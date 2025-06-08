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
@torch.no_grad()
def cast_actions_by_yaw(yaws: torch.tensor, target_yaws: torch.tensor, target_actions: list):
    # yaw increases trigonometrically. upright is 0, turning left increases, right decreases. -pi < yaw < pi.
    rad_distances = torch.minimum(torch.minimum((yaws - target_yaws).abs(), (yaws - 2 * math.pi - target_yaws).abs()), (yaws + 2 * math.pi - target_yaws).abs())
    mask_keep_action = rad_distances < math.pi / 6
    mask_throw_action = rad_distances > 3 * math.pi / 4
    mask_change_action = torch.logical_and(mask_keep_action ^ True, mask_throw_action ^ True)
    mask_to_left = torch.logical_and(
        mask_change_action,
        torch.minimum((target_yaws - yaws).abs(), (target_yaws + 2 * math.pi - yaws).abs()) <
        torch.minimum((yaws - target_yaws).abs(), (yaws + 2 * math.pi - target_yaws).abs())
    )
    mask_to_right = torch.logical_and(mask_change_action, mask_to_left ^ True)

    mask_throw_action = mask_throw_action.numpy()
    mask_change_action = mask_change_action.numpy()
    mask_to_left = mask_to_left.numpy()
    mask_to_right = mask_to_right.numpy()

    target_actions_indexes = [utils.ACTION_INDEX_HT[action] for action in target_actions]  # merge pentru 6 actiuni, cu virajele L, S, R, L, S, R.

    changed_target_actions = copy.deepcopy(target_actions)
    for i in range(len(target_actions)):
        if mask_change_action[i]:
            a_ind = target_actions_indexes[i]
            offset = 3 if a_ind >= 3 else 0
            a_ind -= offset
            a_ind = a_ind - mask_to_left[i] + mask_to_right[i]
            if 0 <= a_ind < 3:
                a_ind += offset
                changed_target_actions[i] = utils.VALUES_ACTIONS[a_ind]
            else:
                mask_throw_action[i] = True

    return changed_target_actions, mask_throw_action
