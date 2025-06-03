import torch.nn.functional as F
import pandas as pd
import numpy as np
import itertools
import random
import torch
import time
import math
import copy
import os

import qnet_conv_helper
import utils

class Agent:
    def __init__(self):
        self.states = []  # the states through which we have gone through during the current episode. (raw format from the game)
        self.actions = []  # the actions that we have chosen during the episode. array of tuples (steer, gas, brake).
        self.rewards = []
        self.agent_wants_new_episode = False

        self.episode_ind = 1  # the id of the current episode. 1-indexed.
        self.dbg_every = 50  # forcefully debug every ??th episode. (e.g. have a run with argmax Q choices and print max(Q(s[0], a) | a).
        self.inf = (1 << 30) * 1.0

        # hyperparameters:
        self.CNT_REPLAYS = len([entry for entry in os.scandir(utils.REPLAYS_DIR) if entry.is_file()])

        self.BATCH_SIZE = 128
        self.Q_LR = 1e-2
        self.LR = 3e-3
        self.REPLAY_BUFSIZE = 3 * 10 ** 3

        self.DISCOUNT_FACTOR = 0.99

        self.SOFTMAX_SCHEDULER = utils.DecayScheduler(start=2, end=0.5, decay=500) # temperature for the softmax policy. (10, 1, 500)
        self.CNT_REPEATING_ACTIONS = utils.DecayScheduler(start=50, end=15, decay=500)  # will choose a new action every CNT_REPEATING_ACTIONS.

        self.REWARD_END_EPISODE_COEF_START = 10.0
        self.REWARD_END_EPISODE_COEF_END = 30.0

        # hyperparameters end.

        self.priority_replay_buffer = qnet_conv_helper.WeightedDeque(maxlen = self.REPLAY_BUFSIZE)  # we only use a prioritised replay buffer.

        # load fixed points:
        self.replays_states_actions = []
        self.rsa_ht = {"x": 0, "y": 1, "z": 2, "yaw": 3, "pitch": 4, "roll": 5, "steer": 6, "gas": 7, "brake": 8, "replay_index": 9, "replay_id": 10}
        for eid, entry in zip(itertools.count(), os.scandir(utils.REPLAYS_DIR)):
            if entry.is_file():
                df = pd.read_csv(entry.path, skipinitialspace = True)
                df = df.astype({c: np.float32 for c in df.select_dtypes(include = "float64").columns})

                self.replays_states_actions.append(df[["x", "y", "z", "yaw", "pitch", "roll", "steer", "gas", "brake"]][1:].to_numpy())

                # append another two columns:
                # * index of the (state, action) in the replay, used when computing the reward. (e.g. a column with 0/1/2/...)
                # * entry id. used when computing the state image. (e.g. a column with eid/eid/eid/...)
                self.replays_states_actions[-1] = np.hstack([
                    self.replays_states_actions[-1],
                    np.arange(len(self.replays_states_actions[-1])).reshape(-1, 1),
                    eid * np.ones((len(self.replays_states_actions[-1]), 1))
                ])

        self.replays_states_actions = np.vstack(self.replays_states_actions)

        # stochastically map all steer values which aren't full left/right or center to one of those.
        for i in range(len(self.replays_states_actions)):
            steer = self.replays_states_actions[i, self.rsa_ht["steer"]]
            if steer not in utils.VALUES_STEER:
                sign = 1 if steer > 0 else -1
                steer *= sign
                steer = utils.VAL_STEER_RIGHT if random.random() < steer / utils.VAL_STEER_RIGHT else 0
                self.replays_states_actions[i, self.rsa_ht["steer"]] = steer * sign

        self.replays_states_actions = torch.from_numpy(self.replays_states_actions)

        # the model's parameters, each fixed point from self.replays_states_actions has an amplitude and a standard deviation.
        self.fps_amp = torch.nn.Parameter(torch.zeros(len(self.replays_states_actions)))
        self.fps_std = torch.nn.Parameter(torch.zeros(len(self.replays_states_actions)))

        # masks for which points represent which action.
        self.fps_masks = torch.stack([
            torch.all(self.replays_states_actions[:, self.rsa_ht["steer"]: self.rsa_ht["steer"] + 3] == torch.tensor(steer_gas_brake), dim=1)
            for steer_gas_brake in utils.VALUES_ACTIONS
        ])

        # debug metrics:
        self.dbg_tstart = time.time()
        self.dbg_log = open(f"{utils.LOG_OUTPUT_DIR_PREFIX}q_fixed_points_{int(self.dbg_tstart)}.txt", "w")
        self.dbg_ht = {"receive_new_state": 0.0, "next_action": 0.0}

        print(f"agent_q_fixed_points loaded.")


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
        self.actions = []
        self.rewards = []
        self.episode_ind += 1

    """
    We call this from episode_ended().
    We have ~2s to run as many batch updates as we can on the DQN from the memory buffer.
    """
    def qlearn_update(self):
        if len(self.priority_replay_buffer) < self.BATCH_SIZE:
            return

        t_start = time.time()
        running_loss, loop_id = 0.0, 0
        while time.time() - t_start < utils.MAX_TIME_INBETWEEN_RUNS:
            sample_indexes = self.priority_replay_buffer.sample(self.BATCH_SIZE)
            samples = [self.priority_replay_buffer.dq_items[index] for index in sample_indexes]

            states = torch.stack([state for state, _, _, _ in samples])
            actions = [action for _, action, _, _ in samples]
            rewards = torch.tensor([reward for _, _, reward, _ in samples])
            next_states = torch.stack([next_state if next_state is not None else torch.zeros_like(state) for state, _, _, next_state in samples])
            next_state_end = torch.tensor([next_state is None for _, _, _, next_state in samples])

            # fp_scores.shape = (BS, FPS) -- batch size, fixed point size.
            fp_scores = -torch.linalg.norm(states.unsqueeze(dim = 1) - self.replays_states_actions[:, :3], dim = -1) / torch.exp(self.fps_std) + self.fps_amp
            action_indexes = torch.tensor([utils.ACTION_INDEX_HT[action] for action in actions])

            q_hat = torch.where(
                self.fps_masks[action_indexes].sum(dim=1) < 1e-10,
                -self.inf,
                torch.where(self.fps_masks[action_indexes], fp_scores, torch.zeros_like(fp_scores)).sum(dim=1) / self.fps_masks[action_indexes].sum(dim=1)
            )

            with torch.no_grad():
                fp_scores_next = -torch.linalg.norm(next_states.unsqueeze(dim = 1) - self.replays_states_actions[:, :3], dim = -1) / torch.exp(self.fps_std) + self.fps_amp

                q_hat_next_max = torch.where(
                    self.fps_masks.sum(dim = -1) < 1e-10,
                    -self.inf + torch.zeros(self.BATCH_SIZE, len(utils.VALUES_ACTIONS)),
                    torch.where(
                        self.fps_masks,
                        fp_scores_next.unsqueeze(dim = 1),
                        torch.zeros(self.BATCH_SIZE, len(utils.VALUES_ACTIONS), len(self.replays_states_actions)) # broadcast all to (BS, |A|, FPS).
                    ).sum(dim = -1) / self.fps_masks.sum(dim = -1)
                ).max(dim = 1).values

                # q_target = torch.where(next_state_end, )
                q_target = (1 - self.Q_LR) * q_hat + self.Q_LR * (rewards + self.DISCOUNT_FACTOR * torch.where(next_state_end, torch.zeros_like(q_hat), q_hat_next_max))

            if self.fps_amp.grad is not None: self.fps_amp.grad.zero_()
            if self.fps_std.grad is not None: self.fps_std.grad.zero_()

            loss = F.smooth_l1_loss(q_hat, q_target)
            with torch.no_grad(): running_loss += loss.item()

            loss.backward()

            with torch.no_grad():
                self.fps_amp.sub_(self.fps_amp.grad, alpha = self.LR)
                self.fps_std.sub_(self.fps_std.grad, alpha = self.LR)

            loop_id += 1

        running_loss /= loop_id

        dbg_str = f"finished {loop_id} batches, mean batch loss = {round(running_loss, 5)}."
        self.dbg_log.write(dbg_str + "\n"); self.dbg_log.flush()
        print(dbg_str)


    """
    Called by the client to let us know that we passed a checkpoint.
    """
    def passed_checkpoint(self):
        self.rewards[-1] += utils.REWARD_COEF_PER_CHECKPOINT * (utils.MAX_TIME // utils.GAP_TIME - (len(self.states) - 1))


    """
    Called by the client to let us know that the episode ended, either normally by finishing the map, or forcefully by us.
    """
    def episode_ended(self, did_episode_end_normally: bool):
        if did_episode_end_normally:
            reward_coef = self.REWARD_END_EPISODE_COEF_START
            if (len(self.states) - 1) * utils.GAP_TIME <= utils.BONUS_TIME_END:
                t = max((len(self.states) - 1) * utils.GAP_TIME, utils.BONUS_TIME_START)
                reward_coef = (self.REWARD_END_EPISODE_COEF_END - (t - utils.BONUS_TIME_START) * (self.REWARD_END_EPISODE_COEF_END - self.REWARD_END_EPISODE_COEF_START) / (utils.BONUS_TIME_END - utils.BONUS_TIME_START))
            self.rewards[-1] += reward_coef * (utils.MAX_TIME // utils.GAP_TIME - (len(self.states) - 1))

        # we put all updates into self.replay_buffer here.
        cnt_repeating_actions = int(self.CNT_REPEATING_ACTIONS.get(self.episode_ind))
        for i in range(0, len(self.actions), cnt_repeating_actions):
            j = i + cnt_repeating_actions

            self.priority_replay_buffer.append(
                item = (self.states[i], self.actions[i], sum(self.rewards[i: j]), self.states[j] if j < len(self.states) else None),
                weight = 1.0
            )

        for dbg_str in [
            f"Episode {self.episode_ind}:",
            f"Rewards: min = {round(min(self.rewards), 3)}, avg = {round(sum(self.rewards) / len(self.rewards), 3)}, max = {round(max(self.rewards), 3)}, sum = {round(sum(self.rewards), 3)}",
            f"Time from start: {round(time.time() - self.dbg_tstart, 3)}, this episode in receive_new_state: {round(self.dbg_ht['receive_new_state'], 3)}, next_action: {round(self.dbg_ht['next_action'], 3)}"
        ]:
            self.dbg_log.write(dbg_str + "\n"); self.dbg_log.flush()
            print(dbg_str)

        self.qlearn_update()

        utils.write_processed_output(
            fname = f"{utils.PARTIAL_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{self.episode_ind}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
            actions = self.actions,
            outstream = None # we don't log partial runs.
        )
        if self.episode_ind % self.dbg_every == 0:
            torch.save({"amp": self.fps_amp, "std": self.fps_std}, f"{utils.QNET_LOAD_PTS_DIR}fps_amp_std_{int(self.dbg_tstart)}_{self.episode_ind}.pt")

        if did_episode_end_normally:
            utils.write_processed_output(
                fname = f"{utils.PROCESSED_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{self.episode_ind}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
                actions = self.actions,
                outstream = self.dbg_log
            )

        self.clear_episode()


    """
    Called by the client to give us a new state. Is called by the client because we either: 
    * called want_new_episode().
    * responded to send_action(), and now we got the resulting next state.
    """
    def receive_new_state(self, state: tuple):
        tstart = time.time()

        self.states.append(torch.tensor(state[:3]))

        is_state_too_bad = False
        # e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        # or too far away from any replay point.
        if len(self.states) * utils.GAP_TIME > utils.MAX_TIME:
            is_state_too_bad = True

        self.dbg_ht["receive_new_state"] += time.time() - tstart

        if is_state_too_bad:
            self.want_new_episode()


    """
    Called by the client to ask for a new action.
    Will add a tuple (steer, gas, brake) to the internal self.actions list.
    The caller will access our internal self.actions list for further use.
    """
    @torch.no_grad()
    def next_action(self):
        tstart = time.time()

        if len(self.actions) % int(self.CNT_REPEATING_ACTIONS.get(self.episode_ind)):
            # just copy the last action.
            self.actions.append(self.actions[-1])
        else:
            # the last state's scores against all fixed points:
            fp_scores = -torch.linalg.norm(self.states[-1] - self.replays_states_actions[:, :3], dim = 1) / torch.exp(self.fps_std) + self.fps_amp

            # we compute for each action the mean estimated q score.
            q_hat = torch.tensor([
                fp_scores[self.fps_masks[i]].sum() / self.fps_masks[i].sum() if self.fps_masks[i].sum() > 0 else -self.inf
                for i in range(len(utils.VALUES_ACTIONS))
            ])

            temp = self.SOFTMAX_SCHEDULER.get(self.episode_ind)
            action_index = F.softmax(q_hat / temp, dim=0).multinomial(num_samples=1).item() if self.episode_ind % self.dbg_every != 0 else q_hat.argmax().item()

            self.actions.append(utils.VALUES_ACTIONS[action_index])

        self.rewards.append(-1)

        # TODO: reward pasiv pentru distanta parcursa + reward activ daca am ajuns intr-un punct mai repede decat replay-urile.
        # TODO: mai putine actiuni posibile.

        self.dbg_ht["next_action"] += time.time() - tstart
