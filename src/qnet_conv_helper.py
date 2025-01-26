from collections import deque, namedtuple
from operator import attrgetter
import numpy as np
import itertools
import random
import typing
import bisect
import torch

import utils

class DQN(torch.nn.Module):
    # (for now) the state is represented by the 3D minimap (X = 19, Y = 9, Z = 19) of the surroundings of the car, +Z represents its walking direction.
    # the network outputs the estimates for Q(s, a) for all 12 possible actions (3 for now, only for STEER).
    def __init__(self):
        super(DQN, self).__init__()

        self.imsize = (19, 9, 19)

        # TODO: more out channels, pooling? revert from utils.VALUES_STEER to utils.VALUES_ACTIONS.
        # TODO: 2/3 consecutive states as Transition input? GAP?
        self.conv = torch.nn.Conv3d(in_channels = 1, out_channels = 12, kernel_size = (3, 3, 3))
        self.pool = torch.nn.MaxPool3d(kernel_size = (2, 2, 2))
        self.conv_relu = torch.nn.ReLU()

        fc_insize = np.prod([(ims - ks_c + 1) // ks_p for ims, ks_c, ks_p in zip(self.imsize, self.conv.kernel_size, self.pool.kernel_size)])

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(fc_insize, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, len(utils.VALUES_STEER))
        )

    # state_im should have a shape of [batch_size, in_channels = 1, *self.imsize].
    def forward(self, state_im: torch.tensor):
        # conv(x)'s shape is [batch_size, out_channels, ??, ??, ??], channel avg conv <=> mean on dim 1. we view to flatten everything but the batch size.
        out = self.conv_relu(self.pool(self.conv(state_im)).mean(dim = 1).view(state_im.shape[0], -1))
        out = self.fc_layers(out)
        return out


class DQN2(torch.nn.Module):
    # num_state_ims: how many consecutive state images we receive at input.
    # this has two conv layers, potentially multiple consecutive state images as input, as well as GAP.
    def __init__(self, num_state_ims):
        super(DQN2, self).__init__()

        self.num_state_ims = num_state_ims
        self.imsize = (19, 9, 19)

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels = num_state_ims, out_channels = 12, kernel_size = (3, 3, 3)), # (8, 3, 8) per canal.
            torch.nn.MaxPool3d(kernel_size = (2, 2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels = 12, out_channels = 24, kernel_size = (3, 3, 3)), # (6, 1, 6) per canal.
            torch.nn.ReLU()
        )

        fc_insize = np.prod([
            (ims - ks_c1 + 1) // ks_p - ks_c2 + 1
            for ims, ks_c1, ks_p, ks_c2 in zip(self.imsize, self.conv_layers[0].kernel_size, self.conv_layers[1].kernel_size, self.conv_layers[3].kernel_size)
        ]) * self.conv_layers[3].out_channels

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(fc_insize, 32),
            torch.nn.ReLU()
        )

        self.fc_last_layer = torch.nn.Linear(32 + self.conv_layers[3].out_channels, len(utils.VALUES_STEER))

    # state_im should have a shape of [batch_size, in_channels = self.num_state_ims, *self.imsize].
    def forward(self, state_im: torch.tensor):
        out_conv = self.conv_layers(state_im) # out_conv's shape is [batch_size, out_channels = 24, 6, 1, 6].
        out = self.fc_layers(out_conv.view(state_im.shape[0], -1)) # out's shape is [batch_size, 32].
        return self.fc_last_layer(torch.hstack([out, out_conv.mean(dim = [-3, -2, -1])])) # append the means from all second layer kernels.


class Transition:
    # if a sequence of state_ims are used as the input state, batch the sequence together by increasing the number of channels for self.state_im.
    # here, state_ims are kept as [num_channels, ??, ??, ??], without the batch component, which is added at communication time with the DQN.
    def __init__(self, state_im: torch.tensor, action: tuple, reward: float, next_state_im: torch.tensor):
        self.state_im = state_im
        self.action = action
        self.reward = reward
        self.next_state_im = next_state_im


class TopBuffer:
    # keeps transitions from the top ?? episodes.
    def __init__(self, max_num_episodes: int):
        self.EpisodeInfo = namedtuple("EpisodeInfo", "length index")
        self.max_num_episodes = max_num_episodes
        self.kept_episode_lengths_indexes = []
        self.episode_ht = {} # index: episode_transitions.
        self.curr_index = 0

    def add_episode(self, episode_len: int, episode_transitions: typing.List[Transition]):
        if len(self.kept_episode_lengths_indexes) >= self.max_num_episodes and self.kept_episode_lengths_indexes[-1].length <= episode_len:
            return

        if len(self.kept_episode_lengths_indexes) >= self.max_num_episodes:
            del self.episode_ht[self.kept_episode_lengths_indexes[-1].index]
            self.kept_episode_lengths_indexes.pop()

        self.episode_ht[self.curr_index] = episode_transitions
        bisect.insort(self.kept_episode_lengths_indexes, self.EpisodeInfo(episode_len, self.curr_index), key = attrgetter("length"))
        self.curr_index += 1

    def sample(self, num_samples: int):
        if num_samples <= 0:
            return

        total_len = sum([len(self.episode_ht[index]) for index in self.episode_ht])
        chosen_indexes = list(range(total_len))
        random.shuffle(chosen_indexes)
        chosen_indexes = sorted(chosen_indexes[:num_samples], reverse = True)

        transitions = []
        total_len = 0
        for index in self.episode_ht:
            for i, trans in zip(itertools.count(), self.episode_ht[index]):
                if chosen_indexes and chosen_indexes[-1] - total_len == i:
                    transitions.append(trans)
                    chosen_indexes.pop()
            total_len += len(self.episode_ht[index])

        return transitions


class WeightedDeque:
    # this circular deque remembers pairs of (item, weight sum).
    # if a pair's weight is w, and the total weight of the items in the deque is sw, then that item is sampled (with replacement) with proba w / sw.
    # this class supports weight updates.
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.dq_items = [None for _ in range(self.maxlen)]
        self.dq_aib_weights = [0.0 for _ in range(self.maxlen)]  # this is the Binary Indexed Tree version of the weights array.
        self.tail = 0  # the position at which to add a new element.
        self.len = 0
        self.maxpas = 1
        while self.maxpas < self.maxlen:
            self.maxpas *= 2

    def __len__(self):
        return self.len

    def _lsb(self, x: int):
        return x & (-x)

    def _aib_prefsum(self, index: int):
        ans, i = 0, index + 1
        while i > 0:
            ans += self.dq_aib_weights[i - 1]
            i -= self._lsb(i)
        return ans

    def update_weight(self, index: int, new_weight: float):
        diff_weight, i = new_weight - (self._aib_prefsum(index) - self._aib_prefsum(index - 1)), index + 1 # compute the difference between the new and old weight.
        while i <= self.maxlen:
            self.dq_aib_weights[i - 1] += diff_weight
            i += self._lsb(i)

    def append(self, item, weight: float):
        self.dq_items[self.tail] = item
        self.update_weight(self.tail, weight)
        self.tail = (self.tail + 1) % self.maxlen
        self.len = min(self.maxlen, self.len + 1)

    def extend(self, items_weights: list):
        for item, weight in items_weights:
            self.append(item, weight)

    # returns the self.dq_items indexes of the chosen objects.
    def sample(self, batch_size: int):
        if self.len == 0:
            return []

        indexes = []
        for _ in range(batch_size):
            r = random.random() * self._aib_prefsum(self.maxlen - 1)

            # we search for the rightmost index for which prefsum(index - 1) <= r.
            index, pas = 0, self.maxpas
            while pas:
                if index + pas < self.maxlen and self._aib_prefsum(index + pas - 1) <= r:
                    index += pas
                pas >>= 1
            indexes.append(index)

        return indexes
