from sklearn.neighbors import KDTree
from tinygrad import Tensor, nn
import pandas as pd
import numpy as np
import random
import time
import os

import utils

class QNet:
    def __init__(self, s_mean: np.array, s_std: np.array):
        self.l1 = nn.Linear(10, 32, bias = True)
        self.l2 = nn.Linear(32, 1, bias = True)
        self.s_mean = s_mean
        self.s_std = s_std

    def __call__(self, s: tuple, steer, gas, brake, requires_grad) -> Tensor:
        x = Tensor([(s[0] - self.s_mean[0]) / self.s_std[0], (s[1] - self.s_mean[1]) / self.s_std[1], (s[2] - self.s_mean[2]) / self.s_std[2],
                    1.0 if steer == utils.VAL_STEER_LEFT else 0.0, 1.0 if steer == utils.VAL_NO_STEER else 0.0, 1.0 if steer == utils.VAL_STEER_RIGHT else 0.0,
                    1.0 if steer == utils.VAL_NO_GAS else 0.0, 1.0 if steer == utils.VAL_GAS else 0.0,
                    1.0 if steer == utils.VAL_NO_BRAKE else 0.0, 1.0 if steer == utils.VAL_BRAKE else 0.0], requires_grad = requires_grad)

        x = self.l1(x)
        x = x.relu()
        x = self.l2(x)

        return x

    def compute_best_action_q(self, s: tuple):
        best_q, best_action = None, None
        for steer in utils.VALUES_STEER:
            action = steer, utils.VAL_GAS, utils.VAL_NO_BRAKE
            q = self.__call__(s, *action, requires_grad = False).numpy()[0]
            if best_q is None or q > best_q:
                best_action, best_q = action, q
        return best_action, best_q


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

        self.qnet = QNet(
            s_mean = np.array([np.float32(self.replays_states_actions[:, i].mean()) for i in range(3)]),
            s_std = np.array([np.float32(self.replays_states_actions[:, i].std()) for i in range(3)])
        )

        self.qnet_optimizer = nn.optim.SGD([self.qnet.l1.weight, self.qnet.l1.bias, self.qnet.l2.weight, self.qnet.l2.bias], lr = 1e-3)

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

        t1 = time.time()
        for i in range(len(self.actions[0]) - 1, -1, -1):
            s, a = self.states[i], (self.actions[utils.IND_STEER][i], self.actions[utils.IND_GAS][i], self.actions[utils.IND_BRAKE][i])

            q = self.qnet(s, *a, requires_grad = True)  # the current Q(s, a).

            if i + 1 == len(self.actions[0]):
                last_reward = int(utils.MAX_TIME // utils.GAP_TIME) - (len(self.states) - 1) if did_episode_end_normally else 0
                expected_q = Tensor([np.float32(last_reward)])  # Q(s, a) <- (1 - lr) * Q(s, a) + lr * last_reward
            else:
                # compute r(s, a):
                z = np.exp(- np.linalg.norm(self.replays_states_actions[indexes[i], :3] - s, axis = 1) * self.REWARD_SPREAD_SQ_2X_INV)
                r = -1 + self.REWARD_COEF * sum([
                    z[j] for j in range(self.TOPK_CLOSEST) if self.visited_replay_state[indexes[i, j]] == i and
                                                              all(self.replays_states_actions[indexes[i, j]][3:] == a)
                ]) / self.TOPK_CLOSEST

                expected_q = Tensor([np.float32(r + self.DISCOUNT_FACTOR * self.qnet.compute_best_action_q(self.states[i+1])[1])])

            t2 = time.time()
            print(f"{round(t2 - t1, 3)}, ", end = '')
            t1 = t2

            with Tensor.train():
                loss = 0.5 * ((q - expected_q) ** 2).mean()
                self.qnet_optimizer.zero_grad()
                loss.backward()
                self.qnet_optimizer.step()

            t2 = time.time()
            print(f"{round(t2 - t1, 3)}")
            t1 = t2

        self.LR *= self.RATE_UPD
        self.EPSILON *= self.RATE_UPD

        if self.episode_ind % self.dbg_every == 0:
            print(f"{round(time.time() - self.dbg_tstart, 3)}s, {self.episode_ind = }, max(Q(s[0], a) | a) = {round(self.qnet.compute_best_action_q(self.states[0])[1], 3)}.")

            utils.write_processed_output(
                fname = f"{utils.PARTIAL_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{self.episode_ind}.txt",
                actions = self.actions,
                mention_write = False
            )


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
        self.states.append((state[utils.IND_X], state[utils.IND_Y], state[utils.IND_Z]))

        is_state_too_bad = False  # e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        if len(self.states) * utils.GAP_TIME >= utils.MAX_TIME:
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
            if random.random() < self.EPSILON and self.episode_ind % self.dbg_every:
                best_steer = random.choice(utils.VALUES_STEER)
                best_gas = utils.VAL_GAS # random.choice(utils.VALUES_GAS)
                best_brake = utils.VAL_NO_BRAKE # random.choice(utils.VALUES_BRAKE)
            else:
                best_steer, best_gas, best_brake = self.qnet.compute_best_action_q(self.states[-1])[0]

            self.actions[utils.IND_STEER].append(best_steer)
            self.actions[utils.IND_GAS].append(best_gas)
            self.actions[utils.IND_BRAKE].append(best_brake)
