from sklearn.neighbors import KDTree
import jax.numpy as jnp
import pandas as pd
import numpy as np
import random
import time
import jax
import os

import utils

def qnet_init_params():
    def random_layer_params(l, r, key, scale=1e-2):
        w_key, b_key = jax.random.split(key)
        return scale * jax.random.normal(w_key, (l, r)), scale * jax.random.normal(b_key, (r,))

    layer_sizes = [3 + 3 + 2 + 2, 32, 1]
    return [random_layer_params(l, r, k) for l, r, k in zip(layer_sizes[:-1], layer_sizes[1:], jax.random.split(jax.random.key(0), len(layer_sizes)))]

@jax.jit
def qnet_forward(params, s_mean: jnp.array, s_std: jnp.array, s, a):
    x = jnp.array([
        *(jnp.array(s) - s_mean) / s_std,
        jnp.where(a[utils.IND_STEER] == utils.VAL_STEER_LEFT, 1.0, 0.0),
        jnp.where(a[utils.IND_STEER] == utils.VAL_NO_STEER, 1.0, 0.0),
        jnp.where(a[utils.IND_STEER] == utils.VAL_STEER_RIGHT, 1.0, 0.0),
        jnp.where(a[utils.IND_GAS] == utils.VAL_NO_GAS, 1.0, 0.0),
        jnp.where(a[utils.IND_GAS] == utils.VAL_GAS, 1.0, 0.0),
        jnp.where(a[utils.IND_BRAKE] == utils.VAL_NO_BRAKE, 1.0, 0.0),
        jnp.where(a[utils.IND_BRAKE] == utils.VAL_BRAKE, 1.0, 0.0)
    ])

    return jnp.dot(jax.nn.relu(jnp.dot(x, params[0][0]) + params[0][1]), params[1][0]) + params[1][1]

@jax.jit
def qnet_loss_fn(params, s_mean: jnp.array, s_std: jnp.array, s, a, y):
    yhat = qnet_forward(params, s_mean, s_std, s, a)
    y = jnp.array([y])
    return 0.5 * jnp.mean((yhat - y) ** 2)

@jax.jit
def qnet_loss_backward(params, s_mean: jnp.array, s_std: jnp.array, lr: jnp.float32, s, a, y):
    loss, grads = jax.value_and_grad(qnet_loss_fn, argnums = 0)(params, s_mean, s_std, s, a, y)
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)], loss

@jax.jit
def jit_compute_best_action_q(params, s_mean: jnp.array, s_std: jnp.array, s):
    s = (jnp.array(s) - s_mean) / s_std

    x = jax.lax.stop_gradient(jnp.vstack([
        [*s, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [*s, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [*s, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    ]))

    return jnp.dot(jax.nn.relu(jnp.dot(x, params[0][0]) + params[0][1]), params[1][0]) + params[1][1]

def compute_best_action_q(params, s_mean: jnp.array, s_std: jnp.array, s):
    q = jit_compute_best_action_q(params, s_mean, s_std, s)
    best_ind = q.argmax().item()
    return utils.VALUES_STEER[best_ind], q[best_ind].item()

class Agent:
    def __init__(self):
        self.states = []  # the states through which we have gone through during the current episode.
        self.actions = ([], [], [])  # the actions that we have chosen during the episode.
        self.agent_wants_new_episode = False

        self.episode_ind = 1  # the id of the current episode. 1-indexed.
        self.dbg_every = 50  # forcefully debug every ??th episode. (e.g. have a run with argmax Q choices and print max(Q(s[0], a) | a).

        # hyperparameters:
        self.CNT_REPLAYS = len([entry for entry in os.scandir(utils.REPLAYS_DIR) if entry.is_file()])
        self.TOPK_CLOSEST = 3 * self.CNT_REPLAYS
        self.REWARD_SPREAD = 3.0  # the smaller the spread, the closer the agent needs to be to a point to get the same reward.
        self.REWARD_SPREAD_SQ_2X_INV = 1 / (2 * self.REWARD_SPREAD ** 2)

        self.RATE_UPD = 1 - 1e-3
        self.LR = 0.9  # Q-learning rate. lr *= rate_upd after each episode. TODO mai e nevoie la optimizer?
        self.REWARD_COEF = 10.0  # r(s, a) = -1 + REWARD_COEF * f(s, a).
        self.DISCOUNT_FACTOR = 0.995
        self.EPSILON = 0.9  # epsilon greedy policy. eps *= rate_upd after each episode.

        self.CNT_REPEATING_ACTIONS = 20 # will choose a new action every CNT_REPEATING_ACTIONS.
        # hyperparameters end.

        self.replays_states_actions = []
        for entry in os.scandir(utils.REPLAYS_DIR):
            if entry.is_file():
                df = pd.read_csv(entry.path, skipinitialspace = True)
                df = df.astype({c: np.float32 for c in df.select_dtypes(include='float64').columns})

                self.replays_states_actions.append(df[['x', 'y', 'z', 'steer', 'gas', 'brake']][1:].to_numpy())

        self.replays_states_actions = np.vstack(self.replays_states_actions)

        for i in range(self.replays_states_actions.shape[0]):  # stochastically map all steer values which aren't full left/right or center to one of those.
            steer = self.replays_states_actions[i, 3 + utils.IND_STEER]
            if steer not in utils.VALUES_STEER:
                sign = 1 if steer > 0 else -1
                steer *= sign
                steer = utils.VAL_STEER_RIGHT if random.random() < steer / utils.VAL_STEER_RIGHT else 0
                self.replays_states_actions[i, 3 + utils.IND_STEER] = steer * sign

        self.kdt = KDTree(self.replays_states_actions[:, :3])
        self.visited_replay_state = np.zeros(self.replays_states_actions.shape[0], dtype = np.int32)  # will not count towards a reward the same replay state twice in the same episode.
        self.visited_replay_state.fill(-1)

        self.s_mean = jnp.array([np.float32(self.replays_states_actions[:, i].mean()) for i in range(3)])
        self.s_std = jnp.array([np.float32(self.replays_states_actions[:, i].std()) for i in range(3)])
        self.qnet_params = qnet_init_params()

        self.dbg_tstart = time.time()

        print(f"agent_qnet loaded.")


    """
    Signal to the client that we want to begin a new episode. We may call this in the other two functions if we:
    * finished an episode by completing the map.
    * or no longer want to explore the current episode.
    """
    def want_new_episode(self):
        self.agent_wants_new_episode = True

    """
    We call this.
    """
    def clear_episode(self):
        self.states = []
        self.actions = ([], [], [])
        self.episode_ind += 1
        self.visited_replay_state.fill(-1)

    """
    We call this either from episode_ended() with did_episode_end_normally = True, or from receive_new_state(), after cancelling an episode with did_episode_end_normally = False.
    This should compute intermediate rewards and update the Q table.
    """
    def qlearn_update(self, did_episode_end_normally: bool):
        _, indexes = self.kdt.query(self.states[: len(self.actions[0])-1], k = self.TOPK_CLOSEST)

        for i in range(len(self.actions[0]) - 1):
            for j in range(self.TOPK_CLOSEST):
                if self.visited_replay_state[indexes[i, j]] == -1:
                    self.visited_replay_state[indexes[i, j]] = i

        running_loss = 0.0
        for i in range(len(self.actions[0]) - 1, -1, -1):
            s, a = self.states[i], (self.actions[utils.IND_STEER][i], self.actions[utils.IND_GAS][i], self.actions[utils.IND_BRAKE][i])

            if i + 1 == len(self.actions[0]):
                last_reward = int(utils.MAX_TIME // utils.GAP_TIME) - (len(self.states) - 1) if did_episode_end_normally else 0
                expected_q = np.float32(last_reward)  # Q(s, a) <- (1 - lr) * Q(s, a) + lr * last_reward
            else:
                # compute r(s, a):
                z = np.exp(- np.linalg.norm(self.replays_states_actions[indexes[i], :3] - s, axis = 1) * self.REWARD_SPREAD_SQ_2X_INV)
                r = -1 + self.REWARD_COEF * sum([
                    z[j] for j in range(self.TOPK_CLOSEST) if self.visited_replay_state[indexes[i, j]] == i and
                                                              all(self.replays_states_actions[indexes[i, j]][3:] == a)
                ]) / self.TOPK_CLOSEST

                expected_q = np.float32(r + self.DISCOUNT_FACTOR * compute_best_action_q(self.qnet_params, self.s_mean, self.s_std, self.states[i+1])[1])

            # We compute the current Q(s, a) below as well.
            self.qnet_params, loss = qnet_loss_backward(self.qnet_params, self.s_mean, self.s_std, 1e-3, s, a, expected_q)
            running_loss += loss

        running_loss /= len(self.actions[0])
        print(f"avg running_loss = {round(running_loss, 3)}")

        self.LR *= self.RATE_UPD
        self.EPSILON *= self.RATE_UPD

        if self.episode_ind % self.dbg_every == 0:
            maxq_s0 = compute_best_action_q(self.qnet_params, self.s_mean, self.s_std, self.states[0])[1]
            print(f"{round(time.time() - self.dbg_tstart, 3)}s, {self.episode_ind = }, max(Q(s[0], a) | a) = {round(maxq_s0, 3)}.")

            utils.write_processed_output(
                fname = f"{utils.PARTIAL_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{self.episode_ind}.txt",
                actions = self.actions,
                mention_write = False
            )

            with open(f"{utils.QNET_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{self.episode_ind}.txt", 'w') as fout:
                fout.write(f"{self.qnet_params}\n")


    """
    Called by the client to let us know that the episode ended, either normally by finishing the map, or forcefully by us.
    """
    def episode_ended(self, did_episode_end_normally: bool):
        self.qlearn_update(did_episode_end_normally)

        if did_episode_end_normally:
            utils.write_processed_output(
                fname = f"{utils.PROCESSED_OUTPUT_DIR_PREFIX}{str(time.time()).replace('.', '')}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
                actions = self.actions,
                mention_write = True
            )

        self.clear_episode()


    """
    Called by the client to give us a new state. Is called by the client because we either: 
    * called want_new_episode().
    * responded to send_action(), and now we got the resulting next state.
    """
    def receive_new_state(self, state: tuple):
        self.states.append(([state[utils.IND_X], state[utils.IND_Y], state[utils.IND_Z]]))

        is_state_too_bad = False  # e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        if len(self.states) * utils.GAP_TIME > utils.MAX_TIME:
            is_state_too_bad = True

        if is_state_too_bad:
            self.want_new_episode()


    """
    Called by the client to ask for a new action.
    Will add a tuple (steer, gas, brake) to the internal self.actions list.
    The caller will access our internal self.actions list for further use.
    """
    def next_action(self):
        if len(self.actions[0]) % self.CNT_REPEATING_ACTIONS:
            # just copy the last action.
            for ind in [utils.IND_STEER, utils.IND_GAS, utils.IND_BRAKE]:
                self.actions[ind].append(self.actions[ind][-1])
        else:
            best_gas = utils.VAL_GAS  # random.choice(utils.VALUES_GAS)
            best_brake = utils.VAL_NO_BRAKE  # random.choice(utils.VALUES_BRAKE)

            if random.random() < self.EPSILON and self.episode_ind % self.dbg_every:
                best_steer = random.choice(utils.VALUES_STEER)
            else:
                best_steer = compute_best_action_q(self.qnet_params, self.s_mean, self.s_std, self.states[-1])[0]

            self.actions[utils.IND_STEER].append(best_steer)
            self.actions[utils.IND_GAS].append(best_gas)
            self.actions[utils.IND_BRAKE].append(best_brake)
