import typing


GAP_TIME = 10 # ms, length of one timestep.
PROCESSED_OUTPUT_DIR_PREFIX = "../processed_outputs/output_"
PARTIAL_OUTPUT_DIR_PREFIX = "../qlearning_outputs/partial_"
FIGURES_OUTPUT_DIR_PREFIX = "../figures/"
QNET_OUTPUT_DIR_PREFIX = "../qnet_outputs/qtable_"

IND_STEER, IND_GAS, IND_BRAKE = 0, 1, 2

VAL_STEER_LEFT, VAL_NO_STEER, VAL_STEER_RIGHT = -65536, 0, 65536
VALUES_STEER = [VAL_STEER_LEFT, VAL_NO_STEER, VAL_STEER_RIGHT]

VAL_NO_GAS, VAL_GAS = 0, 1
VALUES_GAS = [VAL_NO_GAS, VAL_GAS]

VAL_NO_BRAKE, VAL_BRAKE = 0, 1
VALUES_BRAKE = [VAL_NO_BRAKE, VAL_BRAKE]

IND_X, IND_Y, IND_Z, IND_YAW, IND_PITCH, IND_ROLL, IND_VX, IND_VY, IND_VZ, IND_WHEEL_MATERIALS, IND_WHEEL_CONTACT = range(11)

MAX_TIME = 15_000  # maximum number of milliseconds we're willing to run a replay for.
REPLAYS_DIR = "C:\\Users\\ulmea\\Desktop\\Probleme\\Trackmania\\test_date_roti\\RawDataset\\ABC\\A-9_keyboard"


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
"""
def write_processed_output(fname: str, actions: typing.Tuple[typing.List, typing.List, typing.List], mention_write: bool):
    if mention_write:
        print(f"(write_processed_output) Will write to {fname = }.")

    with open(fname, "w") as fout:
        steer, gas, brake = actions[IND_STEER], actions[IND_GAS], actions[IND_BRAKE]
        assert(len(steer) == len(gas) and len(gas) == len(brake))
        n = len(steer)

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
