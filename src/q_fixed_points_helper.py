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
def cast_actions_by_yaw(yaw: torch.tensor, target_yaws: torch.tensor, target_actions: torch.tensor):
    # yaw increases trigonometrically. upright is 0, turning left increases, right decreases. -pi < yaw < pi.
    rad_distances = torch.minimum(torch.minimum((yaw - target_yaws).abs(), (yaw - 2 * math.pi - target_yaws).abs()), (yaw + 2 * math.pi - target_yaws).abs())
    mask_keep_action = rad_distances < math.pi / 6
    mask_throw_action = rad_distances > 3 * math.pi / 4
    mask_change_action = torch.logical_and(mask_keep_action ^ True, mask_throw_action ^ True)
    mask_to_left = torch.logical_and(
        mask_change_action,
        ((target_yaws < yaw) * (target_yaws + 2 * math.pi - yaw) + (target_yaws >= yaw) * torch.minimum(target_yaws - yaw, target_yaws + 2 * math.pi - yaw)) < math.pi
    )
    mask_to_right = torch.logical_and(mask_change_action, mask_to_left ^ True)

    target_actions_indexes = [utils.ACTION_INDEX_HT[tuple(action.tolist())] for action in target_actions]  # merge pentru 6 actiuni, cu virajele L, S, R, L, S, R.
    target_actions = target_actions.clone().numpy()

    for i in range(len(target_actions)):
        if mask_change_action[i]:
            a_ind = target_actions_indexes[i]
            offset = 3 if a_ind >= 3 else 0
            a_ind -= offset
            a_ind = min(2, max(0, a_ind - mask_to_left[i].item() + mask_to_right[i].item()))
            a_ind += offset
            target_actions[i] = utils.VALUES_ACTIONS[a_ind]

    return torch.from_numpy(target_actions), mask_throw_action
