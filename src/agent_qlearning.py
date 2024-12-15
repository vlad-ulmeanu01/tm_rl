from sklearn.neighbors import KDTree
import pandas as pd
import numpy as np
import random
import time
import os

import utils


class Agent:
    def __init__(self):
        self.states = []  # the states through which we have gone through during the current episode.
        self.actions = ([], [], [])  # the actions that we have chosen during the episode.
        self.agent_wants_new_episode = False

        self.episode_ind = 0  # the id of the current episode. 1-indexed.
        self.dbg_every = 25  # forcefully debug every ??th episode. (e.g. have a run with argmax Q choices and print max(Q(s[0], a) | a).

        # hyperparameters:
        self.CNT_REPLAYS = len([entry for entry in os.scandir(utils.REPLAYS_DIR) if entry.is_file()])
        self.TOPK_CLOSEST = 3 * self.CNT_REPLAYS
        self.REWARD_SPREAD_SQ_2X_INV = 1 / (2 * 3.0 ** 2)  # the smaller the spread, the closer the agent needs to be to a point to get the same reward.

        self.RATE_UPD = 1 - 1e-3
        self.LR = 0.9  # Q-learning rate. lr *= rate_upd after each episode.
        self.REWARD_COEF = 0.5  # r(s, a) = -1 + REWARD_COEF * f(s, a).
        self.DISCOUNT_FACTOR = 0.95
        self.EPSILON = 0.9  # epsilon greedy policy. eps *= rate_upd after each episode.

        self.REWARD_ON_FAIL = -int(utils.MAX_TIME // utils.GAP_TIME)  # the reward given when cancelling the episode.
        self.QL_TABLE_CNT_DECIMALS = 1  # how many decimals to keep in the (x, y, z) state representation.
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
        self.q_table = {}  # q_table[(x, y, z)] = [{(steer, gas, brake: best Q value}]

        self.dbg_tstart = time.time()
        print(f"agent_qlearning loaded.")

        self.want_new_episode()  # call want_new_episode() immediately here.


    """
    Signal to the client that we want to begin a new episode. We may call this in the other two functions if we:
    * finished an episode by completing the map.
    * or no longer want to explore the current episode.
    """
    def want_new_episode(self):
        self.agent_wants_new_episode = True
        self.states = []
        self.actions = ([], [], [])
        self.episode_ind += 1


    """
    We call this either from episode_ended() with did_episode_end_normally = True, or from receive_new_state(), after cancelling an episode with did_episode_end_normally = False.
    This should compute intermediate rewards and update the Q table.
    """
    def qlearn_update(self, did_episode_end_normally: bool):
        # assert len(self.states) == len(self.actions[0]) + 1, f"{self.episode_ind = }, {len(self.states) = } != {len(self.actions[0]) + 1 = }."

        for i in range(len(self.actions) - 1, -1, -1):
            s, a = self.states[i], (self.actions[utils.IND_STEER][i], self.actions[utils.IND_GAS][i], self.actions[utils.IND_BRAKE][i])
            if s not in self.q_table:
                self.q_table[s] = {}

            q = self.q_table[s].get(a, 0.0)  # the current Q(s, a).

            if i + 1 == len(self.actions):
                last_reward = 0 if did_episode_end_normally else self.REWARD_ON_FAIL  # all other rewards are -1 + REWARD_COEF * f(s,a).
                q = (1 - self.LR) * q + self.LR * last_reward  # Q(s, a) <- (1 - lr) * Q(s, a) + lr * last_reward
            else:
                # compute r(s, a):
                _, indexes = self.kdt.query([s], k = self.TOPK_CLOSEST)
                indexes = indexes.reshape(-1)

                z = np.exp(- np.linalg.norm(self.replays_states_actions[indexes, :3] - self.states[-1], axis = 1) * self.REWARD_SPREAD_SQ_2X_INV)

                r = -1 + self.REWARD_COEF * sum([z[j] for j in range(self.TOPK_CLOSEST) if all(self.replays_states_actions[j][3:] == a)]) / self.TOPK_CLOSEST

                q = (1 - self.LR) * q + self.LR * (r + self.DISCOUNT_FACTOR * max(self.q_table[self.states[i+1]].values()))

            q = np.float32(q)
            self.q_table[s][a] = q if a not in self.q_table[s] else max(q, self.q_table[s][a])

        self.LR *= self.RATE_UPD
        self.EPSILON *= self.RATE_UPD

        if self.episode_ind % self.dbg_every == 0:
            print(f"{round(time.time() - self.dbg_tstart, 3)} s, {self.episode_ind = }, max(Q(s[0], a) | a) = {round(max(self.q_table[self.states[0]].values()), 3) if self.q_table[self.states[0]] else '??'}.")


    """
    Called by the client to let us know that the episode ended by finishing the map.
    """
    def episode_ended(self):
        self.qlearn_update(did_episode_end_normally = True)

        utils.write_processed_output(
            fname = f"{utils.PROCESSED_OUTPUT_DIR_PREFIX}{str(time.time()).replace('.', '')}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
            actions = self.actions
        )

        self.want_new_episode()


    """
        Called by the client to give us a new state. Is called by the client because we either: 
        * called want_new_episode().
        * responded to send_action(), and now we got the resulting next state.
        """

    def receive_new_state(self, state: tuple):
        self.states.append((
            round(state[utils.IND_X], self.QL_TABLE_CNT_DECIMALS),
            round(state[utils.IND_Y], self.QL_TABLE_CNT_DECIMALS),
            round(state[utils.IND_Z], self.QL_TABLE_CNT_DECIMALS)
        ))

        is_state_too_bad = False  # e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        if len(self.states) * utils.GAP_TIME >= utils.MAX_TIME:
            is_state_too_bad = True

        if is_state_too_bad:
            self.qlearn_update(did_episode_end_normally = False)
            self.want_new_episode()


    """
    Called by the client to ask for a new action.
    Will add a tuple (steer, gas, brake) to the internal self.actions list.
    The caller will access our internal self.actions list for further use.
    """
    def next_action(self):
        # TODO: argmax run self.episode_ind % self.dbg_every. problema e ca self.states[-1] posibil sa nu existe dc urmezi argmax..
        if random.random() < self.EPSILON or self.states[-1] not in self.q_table:
            best_steer = random.choice(utils.VALUES_STEER)
            best_gas = random.choice(utils.VALUES_GAS)
            best_brake = random.choice(utils.VALUES_BRAKE)
        else:
            best_action, best_q = (utils.VAL_NO_STEER, utils.VAL_GAS, utils.VAL_NO_BRAKE), None
            for action, q in self.q_table[self.states[-1]].items():
                if best_q is None or q > best_q:
                    best_action, best_q = action, q
            best_steer, best_gas, best_brake = best_action

        self.actions[utils.IND_STEER].append(best_steer)
        self.actions[utils.IND_GAS].append(best_gas)
        self.actions[utils.IND_BRAKE].append(best_brake)