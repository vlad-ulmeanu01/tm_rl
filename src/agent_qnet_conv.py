from sklearn.neighbors import KDTree
from collections import deque
import pandas as pd
import numpy as np
import itertools
import random
import torch
import time
import math
import os

import utils


class DQN(torch.nn.Module):
    # (for now) the state is represented by the 3D minimap (X = 19, Y = 9, Z = 19) of the surroundings of the car, +Z represents its walking direction.
    # the network outputs the estimates for Q(s, a) for all 12 possible actions.
    def __init__(self):
        super(DQN, self).__init__()

        self.imsize = (19, 9, 19)

        self.conv = torch.nn.Conv3d(in_channels = 1, out_channels = 4, kernel_size = (3, 3, 3))
        self.conv_relu = torch.nn.ReLU()

        fc_insize = np.product([ims - ks + 1 for ims, ks in zip(self.imsize, self.conv.kernel_size)])

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(fc_insize, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, len(utils.VALUES_ACTIONS))
        )

    # state_im should have a shape of [batch_size, in_channels = 1, *self.imsize].
    def forward(self, state_im: torch.tensor):
        # conv(x)'s shape is [batch_size, out_channels, ??, ??, ??], channel avg conv <=> mean on dim 1. we view to flatten everything but the batch size.
        out = self.conv_relu(self.conv(state_im).mean(dim = 1).view(state_im.shape[0], -1))
        for layer in self.fc_layers:
            out = layer(out)
        return out


class Transition:
    def __init__(self, state_im: torch.tensor, action: tuple, reward: float, next_state_im: torch.tensor):
        self.state_im = state_im
        self.action = action
        self.reward = reward
        self.next_state_im = next_state_im


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
        self.REPLAY_BUFSIZE = 2 * 10 ** 4

        self.REWARD_RADIUS = 1 # only attempt to reward replay points that are this close to the current position.
        self.REWARD_COEF = 0.25
        self.REWARD_MAX_EXPONENT = 5

        self.DISCOUNT_FACTOR = 0.995
        self.EPSILON_START, self.EPSILON_END, self.EPSILON_DECAY = 0.9, 0.05, 500 # epsilon greedy policy.

        self.CNT_REPEATING_ACTIONS = 20 # will choose a new action every CNT_REPEATING_ACTIONS.
        # hyperparameters end.

        self.replays_states_actions = []
        for eid, entry in zip(itertools.count(), os.scandir(utils.REPLAYS_DIR)):
            if entry.is_file():
                df = pd.read_csv(entry.path, skipinitialspace = True)
                df = df.astype({c: np.float32 for c in df.select_dtypes(include='float64').columns})

                self.replays_states_actions.append(df[['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'steer', 'gas', 'brake']][1:].to_numpy())

                # append another two columns:
                # * index of the (state, action) in the replay, used when computing the reward. (e.g. a column with 0/1/2/...)
                # * entry id. used when computing the state image. (e.g. a column with eid/eid/eid/...)
                self.replays_states_actions[-1] = np.hstack([
                    self.replays_states_actions[-1],
                    np.arange(len(self.replays_states_actions[-1])).reshape(-1, 1),
                    eid * np.ones((len(self.replays_states_actions[-1]), 1))
                ])

        self.replays_states_actions = np.vstack(self.replays_states_actions)

        for i in range(self.replays_states_actions.shape[0]):  # stochastically map all steer values which aren't full left/right or center to one of those.
            steer = self.replays_states_actions[i, 6 + utils.IND_STEER]
            if steer not in utils.VALUES_STEER:
                sign = 1 if steer > 0 else -1
                steer *= sign
                steer = utils.VAL_STEER_RIGHT if random.random() < steer / utils.VAL_STEER_RIGHT else 0
                self.replays_states_actions[i, 6 + utils.IND_STEER] = steer * sign

        self.replays_kdt = KDTree(self.replays_states_actions[:, :3])
        self.visited_replay_state = np.zeros(self.replays_states_actions.shape[0], dtype = np.int32)  # will not count towards a reward the same replay state twice in the same episode.
        self.visited_replay_state.fill(-1)

        cent_x_half = (np.logspace(0, 1, 10) - 1) / 9; cent_x = np.concatenate([-cent_x_half[1:][::-1], cent_x_half])
        cent_y = np.linspace(-1, 1, 9)
        cent_z = np.linspace(-1, 1, 19)
        self.mesh_kdt = KDTree([(x, y, z) for x in cent_x for y in cent_y for z in cent_z])

        self.replay_buffer = deque([], maxlen = self.REPLAY_BUFSIZE)

        self.qnet = DQN()
        self.qnet_criterion = torch.nn.SmoothL1Loss()
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters())

        self.dbg_tstart = time.time()

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
        self.episode_ind += 1
        self.visited_replay_state.fill(-1)

    """
    We call this from episode_ended().
    We have ~2s to run as many batch updates as we can on the DQN from the memory buffer.
    """
    def qlearn_update(self):
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return

        t_start = time.time()
        running_loss, loop_id = 0.0, 0
        while time.time() - t_start < utils.MAX_TIME_INBETWEEN_RUNS:
            samples = random.sample(self.replay_buffer, self.BATCH_SIZE)

            state_im_batch = torch.stack([transition.state_im for transition in samples]) # shape: [batch_size, 1, 19, 9, 19].
            next_state_im_batch_nonfinal = torch.stack([transition.next_state_im for transition in samples if transition.next_state_im is not None]) # shape: [<= batch_size, 1, 19, 9, 19].
            state_action_values = self.qnet(state_im_batch) # shape: [batch_size, 12].

            with torch.no_grad():
                next_state_action_values_nonfinal = self.qnet(next_state_im_batch_nonfinal).max(dim = 1).values
                expected_state_action_values = torch.clone(state_action_values).detach()

                nonfinal_id = 0
                for batch_id in range(self.BATCH_SIZE):
                    action_id = utils.ACTION_INDEX_HT[samples[batch_id].action]

                    # we only modify action_id's expected value, because it's the only action that we actually did in the episode.
                    if samples[batch_id].next_state_im is None:
                        expected_state_action_values[batch_id, action_id] = 0
                    else:
                        expected_state_action_values[batch_id, action_id] = samples[batch_id].reward + self.DISCOUNT_FACTOR * next_state_action_values_nonfinal[nonfinal_id]
                        nonfinal_id += 1

            loss = self.qnet_criterion(state_action_values, expected_state_action_values)
            running_loss += loss.item()

            self.qnet_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.qnet.parameters(), 100)
            self.qnet_optimizer.step()

            loop_id += 1

        running_loss /= (self.BATCH_SIZE * loop_id)
        print(f"finished {loop_id} batches, avg loss per action output = {round(running_loss, 3)}.")


    """
    Called by the client to let us know that the episode ended, either normally by finishing the map, or forcefully by us.
    """
    def episode_ended(self, did_episode_end_normally: bool):
        if did_episode_end_normally:
            self.rewards[-1] += utils.MAX_TIME - (len(self.states) - 1)

        # we add the last transition from the episode in the buffer. We mark the next state as None to recall its Q(s, .) as 0.
        if len(self.state_ims) >= 2:
            self.replay_buffer.append(Transition(self.state_ims[-2], self.actions[-1], self.rewards[-1], None))

        self.qlearn_update()

        if self.episode_ind % self.dbg_every == 0:
            torch.save(self.qnet.state_dict(), f"{utils.QNET_OUTPUT_DIR_PREFIX}net_{int(self.dbg_tstart)}_{self.episode_ind}.pt")
            utils.write_processed_output(
                fname = f"{utils.PARTIAL_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{str(time.time()).replace('.', '')}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
                actions = self.actions,
                mention_write = False
            )

        if did_episode_end_normally:
            utils.write_processed_output(
                fname = f"{utils.PROCESSED_OUTPUT_DIR_PREFIX}{int(self.dbg_tstart)}_{str(time.time()).replace('.', '')}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
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
        self.states.append(state)

        # compute the new image state state_im here.
        state_im = torch.zeros(1, *self.qnet.imsize) # 1 = in_channels.

        new_origin = (state[utils.IND_X], state[utils.IND_Y], state[utils.IND_Z], state[utils.IND_YAW], state[utils.IND_PITCH], state[utils.IND_ROLL])

        indexes = self.replays_kdt.query_radius([new_origin[:3]], r = self.POINTS_RADIUS)[0]
        if len(indexes):
            pts_trans = utils.transform_about(self.replays_states_actions[indexes, :3], new_origin)
            pts_replay_tags = self.replays_states_actions[indexes, -1] # need to know for each selected point from which replay was it chosen.

            mask_close_y = (-self.TRANS_DIFF_Y <= pts_trans[:, 1]) & (pts_trans[:, 1] <= self.TRANS_DIFF_Y)
            pts_trans /= self.POINTS_RADIUS
            pts_trans[:, 0] *= -1 # TM: X is inverted.

            mesh_indexes = self.mesh_kdt.query(pts_trans, return_distance = False).reshape(-1) # to what mesh point was each pts_trans matched.
            mesh_visited = np.zeros((*self.qnet.imsize, self.CNT_REPLAYS), dtype = np.bool_)

            for tag, mesh_ind in zip(map(int, pts_replay_tags[mask_close_y]), mesh_indexes[mask_close_y]):
                ind_x = mesh_ind // (self.qnet.imsize[1] * self.qnet.imsize[2])
                ind_y = mesh_ind % (self.qnet.imsize[1] * self.qnet.imsize[2]) // self.qnet.imsize[2]
                ind_z = mesh_ind % (self.qnet.imsize[1] * self.qnet.imsize[2]) % self.qnet.imsize[2]

                if not mesh_visited[ind_x, ind_y, ind_z, tag]:
                    mesh_visited[ind_x, ind_y, ind_z, tag] = True
                    state_im[0, ind_x, ind_y, ind_z] += 1

            state_im /= self.CNT_REPLAYS

        self.state_ims.append(state_im)
        if len(self.actions) > 1:
            # we will put the latest Transition only when we finish the episode, we want to mark the next state as None, and optionally increase the reward.
            self.replay_buffer.append(Transition(self.state_ims[-3], self.actions[-2], self.rewards[-2], self.state_ims[-2]))

        is_state_too_bad = False
        # e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        # or too far away from any replay point.
        if len(self.states) * utils.GAP_TIME > utils.MAX_TIME or len(indexes) == 0:
            is_state_too_bad = True

        if is_state_too_bad:
            self.want_new_episode()


    """
    Called by the client to ask for a new action.
    Will add a tuple (steer, gas, brake) to the internal self.actions list.
    The caller will access our internal self.actions list for further use.
    """
    def next_action(self):
        if len(self.actions) % self.CNT_REPEATING_ACTIONS:
            # just copy the last action.
            self.actions.append(self.actions[-1])
        else:
            # best_gas = utils.VAL_GAS
            # best_brake = utils.VAL_NO_BRAKE

            eps = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * math.exp(-self.episode_ind / self.EPSILON_DECAY)
            if random.random() < eps and self.episode_ind % self.dbg_every:
                best_steer = random.choice(utils.VALUES_STEER)
                best_gas = random.choice(utils.VALUES_GAS)
                best_brake = random.choice(utils.VALUES_BRAKE)
            else:
                with torch.no_grad(): # we need to unsqueeze for batch_size = 1.
                    best_steer, best_gas, best_brake = utils.VALUES_ACTIONS[self.qnet(self.state_ims[-1].unsqueeze(dim = 0))[0].argmax().item()]

            self.actions.append((best_steer, best_gas, best_brake))

        # since we just computed the next action, we can also compute the reward given here as well.
        reward = 0

        if len(self.states) > 1: # constant reward: distance travelled between the last two states.
            reward += np.linalg.norm(np.array(self.states[-1][:3], dtype = np.float32) - np.array(self.states[-2][:3], dtype = np.float32))

        indexes = self.replays_kdt.query_radius([self.states[-1][:3]], r = self.REWARD_RADIUS)[0]
        frame_at_replay = self.replays_states_actions[indexes, -2] # at what frame in the replay does the replay hit the close point.
        for ind, replay_frame in zip(indexes, frame_at_replay):
            if self.visited_replay_state[ind] == -1:
                self.visited_replay_state[ind] = 0
                frame = len(self.states) - 1

                # we reward an agent if it reaches a state faster than a replay. we only reward him at most once per replay point per episode.
                reward += math.exp(min(self.REWARD_MAX_EXPONENT, (replay_frame - frame) * self.REWARD_COEF))

        self.rewards.append(reward)
