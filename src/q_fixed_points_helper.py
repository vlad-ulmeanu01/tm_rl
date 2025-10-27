import numpy as np
import torch
import math
import copy

import utils

@torch.no_grad()
def radian_distance_loss(u: torch.tensor, v: torch.tensor):
    return 0.1 * torch.exp(
        torch.minimum(torch.minimum(
            (u - v).abs(),
            (u - 2 * math.pi - v).abs()),
            (u + 2 * math.pi - v).abs())
    ).sum(dim=1)


# we change the action we perceive from a point if our angle against it is different enough. for a high enough difference, we ignore the fixed point altogether.
def cast_actions_by_yaw(yaw: np.float32, target_yaws: np.array, target_action_indexes: np.array):
    # yaw increases trigonometrically. upright is 0, turning left increases, right decreases. -pi < yaw < pi.
    rad_distances = np.minimum(np.minimum(np.abs(yaw - target_yaws), np.abs(yaw - 2 * math.pi - target_yaws)), np.abs(yaw + 2 * math.pi - target_yaws))
    mask_keep_action = rad_distances < math.pi / 6
    mask_throw_action = rad_distances > 3 * math.pi / 4
    mask_change_action = np.logical_and(mask_keep_action ^ True, mask_throw_action ^ True)
    mask_to_left = np.logical_and(
        mask_change_action,
        ((target_yaws < yaw) * (target_yaws + 2 * math.pi - yaw) + (target_yaws >= yaw) * np.minimum(target_yaws - yaw, target_yaws + 2 * math.pi - yaw)) < math.pi
    )
    mask_to_right = np.logical_and(mask_change_action, mask_to_left ^ True)

    target_action_indexes = copy.deepcopy(target_action_indexes)

    for i in range(len(target_action_indexes)):
        if mask_change_action[i]:
            a_ind = target_action_indexes[i]
            offset = 3 if a_ind >= 3 else 0
            a_ind -= offset
            a_ind = min(2, max(0, a_ind - np.int32(mask_to_left[i]) + np.int32(mask_to_right[i])))
            a_ind += offset
            target_action_indexes[i] = a_ind

    return target_action_indexes, mask_throw_action
