import numpy as np
import itertools
import typing
import math


GAP_TIME = 10 # ms, length of one timestep.
PROCESSED_OUTPUT_DIR_PREFIX = "../processed_outputs/output_"
# PARTIAL_OUTPUT_DIR_PREFIX = "../qnet_conv_outputs/partial_"
PARTIAL_OUTPUT_DIR_PREFIX = "../q_fixed_outputs/partial_"
LOG_OUTPUT_DIR_PREFIX = "../logs/"
FIGURES_OUTPUT_DIR_PREFIX = "../figures/"
# QNET_OUTPUT_DIR_PREFIX = "../qnet_conv_outputs/"
QNET_OUTPUT_DIR_PREFIX = "../q_fixed_outputs/"
# QNET_LOAD_PTS_DIR = "../qnet_conv_saved_pts/"
QNET_LOAD_PTS_DIR = "../q_fixed_saved_pts/"

IND_STEER, IND_GAS, IND_BRAKE = 0, 1, 2

VAL_STEER_LEFT, VAL_NO_STEER, VAL_STEER_RIGHT = -65536, 0, 65536
VALUES_STEER = [VAL_STEER_LEFT, VAL_NO_STEER, VAL_STEER_RIGHT]

VAL_NO_GAS, VAL_GAS = 0, 1
VALUES_GAS = [VAL_NO_GAS, VAL_GAS]

VAL_NO_BRAKE, VAL_BRAKE = 0, 1
VALUES_BRAKE = [VAL_NO_BRAKE, VAL_BRAKE]

SUBACTIONS_NAMES = ["steer", "gas", "brake"]
SUBACTIONS_VALUES = {"steer": VALUES_STEER, "gas": VALUES_GAS, "brake": VALUES_BRAKE}

# VALUES_ACTIONS = list(itertools.product(VALUES_STEER, VALUES_GAS, VALUES_BRAKE))
VALUES_ACTIONS = [
    (VAL_STEER_LEFT, VAL_GAS, VAL_NO_BRAKE), (VAL_NO_STEER, VAL_GAS, VAL_NO_BRAKE), (VAL_STEER_RIGHT, VAL_GAS, VAL_NO_BRAKE),
    (VAL_STEER_LEFT, VAL_NO_GAS, VAL_BRAKE), (VAL_NO_STEER, VAL_NO_GAS, VAL_BRAKE), (VAL_STEER_RIGHT, VAL_NO_GAS, VAL_BRAKE)
]
ACTION_INDEX_HT = {action: i for i, action in zip(itertools.count(), VALUES_ACTIONS)}

IND_X, IND_Y, IND_Z, IND_YAW, IND_PITCH, IND_ROLL, IND_VX, IND_VY, IND_VZ, IND_WHEEL_MATERIALS, IND_WHEEL_CONTACT = range(11)

MAX_TIME_INBETWEEN_RUNS = 0.95 # maximum number of seconds that we can do computing between ending an episode and beginning another.

"""
MAX_TIME: maximum number of milliseconds we're willing to run a replay for.
BONUS_TIME_START: time around what we would want to hit.
BONUS_TIME_END: we award a reward bonus for tracks finishing between BONUS_TIME_START, BONUS_TIME_END.
REWARD_COEF_PER_CHECKPOINT: award ?? * (how much extra reward we would have gotten for finishing the track then, with no extra coefficient).
"""

# A-9:
MAX_TIME = 15_000
BONUS_TIME_START = 6_500
BONUS_TIME_END = 9_000
REWARD_COEF_PER_CHECKPOINT = 1.0
REPLAYS_DIR = "C:\\Users\\ulmea\\Desktop\\Probleme\\Trackmania\\test_date_roti\\RawDataset\\ABC\\A-9_keyboard"

# TAS-Train:
# MAX_TIME = 20_000
# BONUS_TIME_START = 9_000
# BONUS_TIME_END = 12_000
# REWARD_COEF_PER_CHECKPOINT = 1.0
# REPLAYS_DIR = "C:\\Users\\ulmea\\Desktop\\Probleme\\Trackmania\\test_date_roti\\RawDataset\\Others\\TAS-Train1"

# A03-Race:
# MAX_TIME = 60_000 # 27_500
# BONUS_TIME_START = 18_000
# BONUS_TIME_END = 23_000
# REWARD_COEF_PER_CHECKPOINT = 1.0
# REPLAYS_DIR = "C:\\Users\\ulmea\\Desktop\\Probleme\\Trackmania\\test_date_roti\\RawDataset\\Others\\A03-Race"

class DecayScheduler:
    def __init__(self, start: float, end: float, decay: float):
        self.start = start
        self.end = end
        self.decay = decay

    def get(self, episode_ind: int):
        amt = math.exp(-episode_ind / self.decay)
        if self.end < self.start:
            return self.end + (self.start - self.end) * amt
        return self.start + (self.end - self.start) * (1 - amt)


"""
Reads an input file. First line steer, second gas, third brake.
"""
def read_processed_input(fname: str):
    with open(fname) as fin:
        actions = tuple([list(map(int, fin.readline().split())) for _ in range(3)]) #[split, push_up, push_down]
        assert(len(actions[IND_STEER]) == len(actions[IND_GAS]) and len(actions[IND_STEER]) == len(actions[IND_BRAKE]))
    return actions


"""
Writes processed actions in a file such that they may be read by TMInterface.
The actions tuple is made out of three lists of equal size: steer, gas and brake actions.
If outstream is not None, we will write some log information to stdout / the outstream.
"""
def write_processed_output(fname: str, actions: typing.List[typing.Tuple], outstream):
    if outstream is not None:
        dbg_str = f"(write_processed_output) Will write to {fname = }."
        outstream.write(dbg_str + "\n"); outstream.flush()
        print(dbg_str)

    with open(fname, "w") as fout:
        steer, gas, brake = [s for s, g, b in actions], [g for s, g, b in actions], [b for s, g, b in actions]
        n = len(actions)

        for push, direction in [(gas, "up"), (brake, "down")]:
            i = 0
            while i < n:
                while i < n and push[i] == 0:
                    i += 1
                if i < n:
                    j = i
                    while j < n and push[j] == 1:
                        j += 1
                    fout.write(f"{i * GAP_TIME}-{j * GAP_TIME} press {direction}\n")
                    i = j

        for i in range(n):
            fout.write(f"{i * GAP_TIME} steer {steer[i]}\n")


# transforms the points in pts: pts.shape = [n, 3(x, y, z)] s.t. their new_origin = (x, y, z, yaw, pitch, roll).
# in trackmania, -x O +z is the plane, and +y exits from it.
def transform_about(pts, new_origin):
    x, y, z, yaw, pitch, roll = new_origin
    pitch, roll = -pitch, -roll  # pitch and roll increase antitrigonometrically ingame.

    ca, sa = math.cos(-yaw), math.sin(-yaw)  # normally, yaw doesn't change +z, but TM swapped +y and +z.. so the yaw matrix must leave +y unchanged.
    matYaw = np.array([[ca, 0, sa, 0], [0, 1, 0, 0], [-sa, 0, ca, 0], [0, 0, 0, 1]], dtype=np.float32)

    cb, sb = math.cos(-pitch), math.sin(-pitch)  # TM: pitch must leave +x unchanged.
    matPitch = np.array([[1, 0, 0, 0], [0, cb, -sb, 0], [0, sb, cb, 0], [0, 0, 0, 1]], dtype=np.float32)

    cc, sc = math.cos(-roll), math.sin(-roll)  # TM: roll must leave +z unchanged.
    matRoll = np.array([[cc, -sc, 0, 0], [sc, cc, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    matTrans = np.array([[1, 0, 0, -x], [0, 1, 0, -y], [0, 0, 1, -z], [0, 0, 0, 1]], dtype=np.float32)

    mat = matRoll @ matPitch @ matYaw @ matTrans
    return (mat @ np.vstack([pts.T, np.ones(pts.shape[0])]))[:3, :].T

# returns True if a and b (angle measures in radians) are at most diff apart. a and b are in [-math.pi, math.pi].
def radian_distance(a, b, diff):
    return min([abs(a - b), abs(a - 2*math.pi - b), abs(a + 2*math.pi - b)]) < diff
