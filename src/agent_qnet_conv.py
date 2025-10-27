from sklearn.neighbors import KDTree
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
        self.state_ims = [] # the states in the format that we actually use them.
        self.actions = []  # the actions that we have chosen during the episode. array of tuples (steer, gas, brake).
        self.rewards = []
        self.agent_wants_new_episode = False

        self.episode_ind = 1  # the id of the current episode. 1-indexed.
        self.dbg_every = 50  # forcefully debug every ??th episode. (e.g. have a run with argmax Q choices and print max(Q(s[0], a) | a).

        # hyperparameters:
        self.CNT_REPLAYS = len([entry for entry in os.scandir(utils.REPLAYS_DIR) if entry.is_file()])
        self.POINTS_RADIUS = 30 # when computing the state image, only consider points from replays if they are at most POINTS_RADIUS away.
        self.TRANS_DIFF_Y = 10 # post transformation to new origin, only keep points that are at most ?? away from the origin on the Y axis.

        self.BATCH_SIZE = 128
        self.LR = 1e-3 # DQN learning rate.
        self.REPLAY_BUFSIZE = 3 * 10 ** 3
        self.PRIO_BUFF_ALPHA, self.PRIO_BUFF_BETA, self.PRIO_BUFF_EPS = 0.7, 0.5, 1e-7

        # first active reward:
        self.REWARD1_TOPK_CLOSEST = 3 * self.CNT_REPLAYS
        self.REWARD1_SPREAD = utils.DecayScheduler(start = 5.0, end = 1.0, decay = 500)  # the smaller the spread, the closer the agent needs to be to a point to get the same reward.
        self.REWARD1_COEF = 10.0  # r(s, a) = REWARD_COEF * f(s, a).

        # second active reward:
        self.REWARD2_RADIUS = 1 # only attempt to reward replay points that are this close to the current position.
        self.REWARD2_COEF = 0.1
        self.REWARD2_MAX_EXPONENT = 1
        self.REWARD2_MIN_FRAME = 100
        self.REWARD2_YPR_RADIAN_DIFF = math.pi / 24 # will only count a point if the orientation is at most ?? degrees away in any axis.
        self.REWARD2_SCHEDULER = utils.DecayScheduler(start = 0.1, end = 1.0, decay = 500) # this should play a bigger role in later episodes.

        self.PASSIVE_REWARD_MAX_BAD_STREAK = 100
        self.PASSIVE_REWARD_BAD_STREAK_BOUND = 1e-2 # if the car moves less than ?? per frame for 100 consecutive frames, cancel the episode.

        self.REWARD_END_EPISODE_COEF_START = 10.0 # (time - utils.BONUS_TIME_START) / (utils.BONUS_TIME_END - utils.BONUS_TIME_START) == (end - bonus) / (end - start).
        self.REWARD_END_EPISODE_COEF_END = 30.0

        self.DISCOUNT_FACTOR = 0.99
        self.OFFLINE_NET_UPD_COEF = 5e-3 # the offline net tends towards the online net with ?? per episode update.

        # self.EPSILON_SCHEDULER = utils.DecayScheduler(start = 0.9, end = 0.05, decay = 500) # epsilon greedy policy.
        self.SOFTMAX_SCHEDULER = utils.DecayScheduler(start = 50, end = 1, decay = 500) # temperature for the softmax policy.

        self.CNT_REPEATING_ACTIONS = utils.DecayScheduler(start = 50, end = 15, decay = 1000) # will choose a new action every CNT_REPEATING_ACTIONS.

        # hyperparameters end.

        self.passive_reward_bad_streak = 0

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

        for i in range(len(self.replays_states_actions)):  # stochastically map all steer values which aren't full left/right or center to one of those.
            steer = self.replays_states_actions[i, self.rsa_ht["steer"]]
            if steer not in utils.VALUES_STEER:
                sign = 1 if steer > 0 else -1
                steer *= sign
                steer = utils.VAL_STEER_RIGHT if random.random() < steer / utils.VAL_STEER_RIGHT else 0
                self.replays_states_actions[i, self.rsa_ht["steer"]] = steer * sign

        self.replays_kdt = KDTree(self.replays_states_actions[:, [self.rsa_ht["x"], self.rsa_ht["y"], self.rsa_ht["z"]]])

        # will not count towards a reward the same replay state twice in the same episode.
        # need two rows because we use two types of active rewards, the first for choosing similar actions in similar positions,
        # and second for arriving to a replay position faster than the replay did.
        self.visited_replay_state = np.zeros((2, len(self.replays_states_actions)), dtype = np.bool_)

        cent_x_half = (np.logspace(0, 1, 10) - 1) / 9; cent_x = np.concatenate([-cent_x_half[1:][::-1], cent_x_half])
        cent_y = np.linspace(-1, 1, 9)
        cent_z = np.linspace(-1, 1, 19)
        self.mesh_kdt = KDTree([(x, y, z) for x in cent_x for y in cent_y for z in cent_z], leaf_size = 3, metric = "l1")

        self.mesh_id_to_loc = np.zeros((len(cent_x) * len(cent_y) * len(cent_z), 3), dtype = np.int32)
        ind_x, ind_y, ind_z = 0, 0, 0
        for ind in range(len(self.mesh_id_to_loc)):
            self.mesh_id_to_loc[ind] = (ind_x, ind_y, ind_z)
            ind_z += 1
            if ind_z >= len(cent_z):
                ind_z = 0
                ind_y += 1
                if ind_y >= len(cent_y):
                    ind_y, ind_z = 0, 0
                    ind_x += 1

        self.priority_replay_buffer = qnet_conv_helper.WeightedDeque(maxlen = self.REPLAY_BUFSIZE) # we only use a prioritised replay buffer.
        self.max_abs_td = 0.0 # the maximum temporal difference in absolute value, encountered in an update loop.

        self.qnet = qnet_conv_helper.DQN2(num_state_ims = 2)
        self.qnet.load_state_dict(torch.load(f"{utils.QNET_LOAD_PTS_DIR}net_1737843159_4000.pt", weights_only = True))

        self.qnet_offline = copy.deepcopy(self.qnet)

        self.qnet_criterion = torch.nn.SmoothL1Loss()
        self.qnet_criterion_nored = torch.nn.SmoothL1Loss(reduction = 'none')
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr = self.LR)

        self.dbg_tstart = time.time()
        self.dbg_reward_log = open(f"{utils.LOG_OUTPUT_DIR_PREFIX}qnet_conv_{int(self.dbg_tstart)}_rewards.txt", "w")
        self.dbg_log = open(f"{utils.LOG_OUTPUT_DIR_PREFIX}qnet_conv_{int(self.dbg_tstart)}.txt", "w")
        self.dbg_ht = {"receive_new_state": 0.0, "next_action": 0.0, "sum_reward_passive": 0.0, "sum_reward_active1": 0.0, "sum_reward_active2": 0.0}

        print(f"agent_qnet_conv loaded.")


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
        self.state_ims = []
        self.actions = []
        self.rewards = []
        self.passive_reward_bad_streak = 0
        self.visited_replay_state.fill(False)
        self.episode_ind += 1
        self.dbg_ht = {x: 0.0 for x in self.dbg_ht}

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

            state_im_batch = torch.stack([transition.state_im for transition in samples]) # shape: [batch_size, num_channels = 2, 19, 9, 19].

            next_state_im_batch_nonfinal = torch.stack([
                torch.cat([transition.state_im[-1].unsqueeze(dim = 0), transition.next_state_im]) # the resulting state_im: last from start + next. need to unsqueeze to keep the channel dim.
                for transition in samples if transition.next_state_im is not None
            ]) # shape: [<= batch_size, num_channels = 2, 19, 9, 19].

            state_action_values = self.qnet(state_im_batch) # shape: [batch_size, 12] (for now: [batch_size, 3]).

            priority_weights = np.zeros(self.BATCH_SIZE)  # we should give lower importance to samples that are likely to be prioritised from the buffer.
            with torch.no_grad():
                # DQN:
                # next_state_action_values_nonfinal = self.qnet_offline(next_state_im_batch_nonfinal).max(dim = 1).values

                # DDQN: the action a that provides the best Q_off(s[t+1], a) may be overestimated. instead, we take a' that provides the best Q_on(s[t+1], a'),
                # then compute the offline net's estimation of it: Q_off(s[t+1], a').
                online_qnet_argmax_nonfinal_indexes = self.qnet(next_state_im_batch_nonfinal).argmax(dim = 1)
                offline_qnet_nonfinal = self.qnet_offline(next_state_im_batch_nonfinal)

                expected_state_action_values = torch.clone(state_action_values).detach()
                nonfinal_id = 0
                for batch_id in range(self.BATCH_SIZE):
                    # action_id = utils.ACTION_INDEX_HT[samples[batch_id].action]
                    action_id = utils.VALUES_STEER.index(samples[batch_id].action[utils.IND_STEER])

                    # we only modify action_id's expected value, because it's the only action that we actually did in the episode.
                    if samples[batch_id].next_state_im is None:
                        expected_state_action_values[batch_id, action_id] = 0
                    else:
                        abs_td = expected_state_action_values[batch_id, action_id].item() # remember the network's estimate of the SA value.

                        expected_state_action_values[batch_id, action_id] = samples[batch_id].reward + self.DISCOUNT_FACTOR * offline_qnet_nonfinal[
                            nonfinal_id,
                            online_qnet_argmax_nonfinal_indexes[nonfinal_id].item() # DDQN: we use the offline estimation for the best action indicated by the online net.
                        ]
                        nonfinal_id += 1

                        # compute the absolute temporal difference and update the deque weight.
                        abs_td = abs(expected_state_action_values[batch_id, action_id].item() - abs_td)
                        new_weight = (abs_td + self.PRIO_BUFF_EPS) ** self.PRIO_BUFF_ALPHA

                        self.max_abs_td = max(self.max_abs_td, abs_td)
                        self.priority_replay_buffer.update_weight(
                            index = sample_indexes[batch_id],
                            new_weight = new_weight
                        )

                        # divide by the sum of all buffer priorities. we use this to compute the probability of an item being chosen.
                        proba = (abs_td + self.PRIO_BUFF_EPS) / self.priority_replay_buffer._aib_prefsum(self.priority_replay_buffer.maxlen - 1)
                        priority_weights[batch_id] = 1.0 / (len(self.priority_replay_buffer) * proba) ** self.PRIO_BUFF_BETA

                priority_weights /= priority_weights.sum()

            # loss = self.qnet_criterion(state_action_values, expected_state_action_values)
            loss = (self.qnet_criterion_nored(state_action_values, expected_state_action_values) * torch.tensor(priority_weights).view(-1, 1)).mean()
            running_loss += loss.item()

            self.qnet_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.qnet.parameters(), 100)
            self.qnet_optimizer.step()

            online_dict, offline_dict = self.qnet.state_dict(), self.qnet_offline.state_dict()
            for key in online_dict:
                offline_dict[key] = offline_dict[key] * (1 - self.OFFLINE_NET_UPD_COEF) + online_dict[key] * self.OFFLINE_NET_UPD_COEF
            self.qnet_offline.load_state_dict(offline_dict)

            loop_id += 1

        running_loss /= loop_id
        
        dbg_str = f"finished {loop_id} batches, avg loss per action output = {round(running_loss, 5)}."
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
                # (time - utils.BONUS_TIME_START) / (utils.BONUS_TIME_END - utils.BONUS_TIME_START) == (end - bonus) / (end - start)
                t = max((len(self.states) - 1) * utils.GAP_TIME, utils.BONUS_TIME_START)
                reward_coef = (self.REWARD_END_EPISODE_COEF_END - (t - utils.BONUS_TIME_START) * (self.REWARD_END_EPISODE_COEF_END - self.REWARD_END_EPISODE_COEF_START) /
                              (utils.BONUS_TIME_END - utils.BONUS_TIME_START))

            self.rewards[-1] += reward_coef * (utils.MAX_TIME // utils.GAP_TIME - (len(self.states) - 1))

        # we pull all self.replay_buffer updates here.
        cnt_repeating_actions = int(self.CNT_REPEATING_ACTIONS.get(self.episode_ind))
        for i in range(0, len(self.actions), cnt_repeating_actions):
            j = i + cnt_repeating_actions
            z = max(0, i - cnt_repeating_actions)

            self.priority_replay_buffer.append(
                item = qnet_conv_helper.Transition(
                    torch.cat([self.state_ims[z], self.state_ims[i]]), # append two consecutive frames as input <=> input state_im shape is [num_channels = 2, ??, ??, ??].
                    self.actions[i],
                    sum(self.rewards[i: j]), # qnet_conv_helper.decayed_sum(self.rewards[i: j], self.DISCOUNT_FACTOR)
                    self.state_ims[j] if j < len(self.states) else None
                ),
                weight = (self.max_abs_td + self.PRIO_BUFF_EPS) ** self.PRIO_BUFF_ALPHA # we insert the transition with the highest known temporal diff s.t. we have a good chance of actually updating its weight.
            )

        for dbg_str in [
            f"Episode {self.episode_ind}:",
            f"Rewards: min = {round(min(self.rewards), 3)}, avg = {round(sum(self.rewards) / len(self.rewards), 3)}, max = {round(max(self.rewards), 3)}, sum = {round(sum(self.rewards), 3)}",
            f"Rewards: passive: {round(self.dbg_ht['sum_reward_passive'], 3)}, active 1: {round(self.dbg_ht['sum_reward_active1'], 3)}, active 2: {round(self.dbg_ht['sum_reward_active2'], 3)}",
            f"Time from start: {round(time.time() - self.dbg_tstart, 3)}, this episode in receive_new_state: {round(self.dbg_ht['receive_new_state'], 3)}, next_action: {round(self.dbg_ht['next_action'], 3)}"
        ]:
            self.dbg_log.write(dbg_str + "\n"); self.dbg_log.flush()
            print(dbg_str)

        self.dbg_reward_log.write(' '.join(map(str, np.round(self.rewards, 3))) + "\n")
        self.dbg_reward_log.flush()

        self.qlearn_update()

        utils.write_processed_output(
            fname = f"{utils.PARTIAL_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{self.episode_ind}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
            actions = self.actions,
            outstream = None # we don't log partial runs.
        )
        if self.episode_ind % self.dbg_every == 0:
            torch.save(self.qnet.state_dict(), f"{utils.QNET_OUTPUT_DIR_PREFIX}net_{int(self.dbg_tstart)}_{self.episode_ind}.pt")

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

        self.states.append(state)

        # compute the new image state state_im here.
        state_im = np.zeros((1, *self.qnet.imsize), dtype = np.float32) # 1 = in_channels.
        away_from_replays = False

        if len(self.actions) % int(self.CNT_REPEATING_ACTIONS.get(self.episode_ind)) == 0: # we actually have to compute the next action, so state_im matters as input for the network.
            new_origin = (state[utils.IND_X], state[utils.IND_Y], state[utils.IND_Z], state[utils.IND_YAW], state[utils.IND_PITCH], state[utils.IND_ROLL])
            indexes = self.replays_kdt.query_radius([new_origin[:3]], r = self.POINTS_RADIUS)[0]

            if len(indexes):
                pts_trans = utils.transform_about(self.replays_states_actions[np.ix_(indexes, [self.rsa_ht["x"], self.rsa_ht["y"], self.rsa_ht["z"]])], new_origin)
                pts_replay_tags = self.replays_states_actions[indexes, self.rsa_ht["replay_id"]] # need to know for each selected point from which replay was it chosen.

                mask_close_y = (-self.TRANS_DIFF_Y <= pts_trans[:, 1]) & (pts_trans[:, 1] <= self.TRANS_DIFF_Y)
                pts_trans /= self.POINTS_RADIUS
                pts_trans[:, 0] *= -1 # TM: X is inverted.

                if sum(mask_close_y):
                    # TODO this if needs to be (again) sped up.
                    mesh_indexes = self.mesh_kdt.query(pts_trans[mask_close_y], return_distance = False).reshape(-1) # to what mesh point was each pts_trans matched.

                    mesh_visited = np.zeros((*self.qnet.imsize, self.CNT_REPLAYS), dtype = np.bool_)
                    for tag, mesh_ind in zip(map(int, pts_replay_tags[mask_close_y]), mesh_indexes):
                        ind_x, ind_y, ind_z = self.mesh_id_to_loc[mesh_ind]
                        if not mesh_visited[ind_x, ind_y, ind_z, tag]:
                            mesh_visited[ind_x, ind_y, ind_z, tag] = True
                            state_im[0, ind_x, ind_y, ind_z] += 1

                    state_im /= self.CNT_REPLAYS
            else:
                away_from_replays = True

        self.state_ims.append(torch.tensor(state_im))

        is_state_too_bad = False
        # e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        # or too far away from any replay point.
        if len(self.states) * utils.GAP_TIME > utils.MAX_TIME or away_from_replays:
            is_state_too_bad = True

        if self.passive_reward_bad_streak >= self.PASSIVE_REWARD_MAX_BAD_STREAK:
            is_state_too_bad = True

        self.dbg_ht["receive_new_state"] += time.time() - tstart

        if is_state_too_bad:
            self.want_new_episode()


    """
    Called by the client to ask for a new action.
    Will add a tuple (steer, gas, brake) to the internal self.actions list.
    The caller will access our internal self.actions list for further use.
    """
    def next_action(self):
        tstart = time.time()

        if len(self.actions) % int(self.CNT_REPEATING_ACTIONS.get(self.episode_ind)):
            # just copy the last action.
            self.actions.append(self.actions[-1])
        else:
            best_gas = utils.VAL_GAS
            best_brake = utils.VAL_NO_BRAKE

            with torch.no_grad():  # we need to unsqueeze for batch_size = 1.
                qnet_out = self.qnet(torch.cat([self.state_ims[-2 if len(self.state_ims) >= 2 else -1], self.state_ims[-1]]).unsqueeze(dim = 0))[0]

                if self.episode_ind % self.dbg_every:
                    best_steer = utils.VALUES_STEER[
                        torch.nn.functional.softmax(
                            qnet_out / self.SOFTMAX_SCHEDULER.get(self.episode_ind), dim = 0
                        ).multinomial(num_samples = 1).item()
                    ]
                else: # do an argmax run every self.dbg_every episodes.
                    best_steer = utils.VALUES_STEER[qnet_out.argmax().item()]

            # if random.random() < self.EPSILON_SCHEDULER.get(self.episode_ind) and self.episode_ind % self.dbg_every:
            #     best_steer = random.choice(utils.VALUES_STEER)
            #     # best_gas = random.choice(utils.VALUES_GAS)
            #     # best_brake = random.choice(utils.VALUES_BRAKE)
            # else:
            #     with torch.no_grad(): # we need to unsqueeze for batch_size = 1.
            #         # best_steer, best_gas, best_brake = utils.VALUES_ACTIONS[self.qnet(self.state_ims[-1].unsqueeze(dim = 0))[0].argmax().item()]
            #         best_steer = utils.VALUES_STEER[self.qnet(self.state_ims[-1].unsqueeze(dim = 0))[0].argmax().item()]

            self.actions.append((best_steer, best_gas, best_brake))

        # since we just computed the next action, we can also compute the reward given here as well.
        reward0 = 0

        # passive reward: distance travelled between the last two states. only count for X/Z, as we need to prevent falling from the map as a reward.
        if len(self.states) > 1:
            reward0 = np.linalg.norm(np.array(self.states[-1][:3], dtype = np.float32)[[utils.IND_X, utils.IND_Z]] -
                                     np.array(self.states[-2][:3], dtype = np.float32)[[utils.IND_X, utils.IND_Z]])
        self.passive_reward_bad_streak = self.passive_reward_bad_streak + 1 if reward0 < self.PASSIVE_REWARD_BAD_STREAK_BOUND else 0

        # active reward 1: award actions similar to ones taken by replays close by.
        indexes = self.replays_kdt.query([self.states[-1][:3]], k = self.REWARD1_TOPK_CLOSEST, return_distance = False)[0]
        z = np.exp(- np.linalg.norm(self.replays_states_actions[np.ix_(indexes, [self.rsa_ht["x"], self.rsa_ht["y"], self.rsa_ht["z"]])] - self.states[-1][:3], axis = 1)
                   / (2 * self.REWARD1_SPREAD.get(self.episode_ind) ** 2))

        chosen_js = [
            j for j in range(self.REWARD1_TOPK_CLOSEST) if not self.visited_replay_state[0, indexes[j]] and
            all(self.replays_states_actions[indexes[j], [self.rsa_ht["steer"], self.rsa_ht["gas"], self.rsa_ht["brake"]]] == self.actions[-1])
        ]

        reward1 = self.REWARD1_COEF * sum(z[chosen_js]) / self.REWARD1_TOPK_CLOSEST
        self.visited_replay_state[0, indexes[chosen_js]] = True # we only mark as visited the replay points which have the same action as self.actions[-1].

        # active reward 2: award if location is close to replay point, but we got there quicker than the replay.
        reward2 = 0

        indexes = self.replays_kdt.query_radius([self.states[-1][:3]], r = self.REWARD2_RADIUS)[0]
        frame_at_replay = self.replays_states_actions[indexes, self.rsa_ht["replay_index"]]  # at what frame in the replay does the replay hit the close point.
        for ind, replay_frame in zip(indexes, frame_at_replay):
            if replay_frame >= self.REWARD2_MIN_FRAME and not self.visited_replay_state[1, ind] and\
                utils.radian_distance(self.states[-1][utils.IND_YAW], self.replays_states_actions[ind, self.rsa_ht["yaw"]], self.REWARD2_YPR_RADIAN_DIFF) and\
                utils.radian_distance(self.states[-1][utils.IND_PITCH], self.replays_states_actions[ind, self.rsa_ht["pitch"]], self.REWARD2_YPR_RADIAN_DIFF) and\
                utils.radian_distance(self.states[-1][utils.IND_ROLL], self.replays_states_actions[ind, self.rsa_ht["roll"]], self.REWARD2_YPR_RADIAN_DIFF):
                self.visited_replay_state[1, ind] = True
                frame = len(self.states) - 1

                # we reward an agent if it reaches a state faster than a replay. we only reward him at most once per replay point per episode.
                reward2 += math.exp(min(self.REWARD2_MAX_EXPONENT, (replay_frame - frame) * self.REWARD2_COEF))

        reward2 *= self.REWARD2_SCHEDULER.get(self.episode_ind)

        self.rewards.append(reward0 + reward1 + reward2)

        self.dbg_ht["sum_reward_passive"] += reward0
        self.dbg_ht["sum_reward_active1"] += reward1
        self.dbg_ht["sum_reward_active2"] += reward2
        self.dbg_ht["next_action"] += time.time() - tstart
