import random
import time

import utils


class Agent:
    def __init__(self):
        self.states = [] # the states through which we have gone through during the current episode.
        self.actions = ([], [], []) # the actions that we have chosen during the episode.

        self.agent_wants_new_episode = False

        self.want_new_episode() # call want_new_episode() immediately here.

    """
    Signal to the client that we want to begin a new episode. We may call this in the other two functions if we:
    * finished an episode by completing the map.
    * or no longer want to explore the current episode.
    """
    def want_new_episode(self):
        self.agent_wants_new_episode = True
        self.states = []
        self.actions = ([], [], [])


    """
    Called by the client to give us a new state. Is called by the client because we either: 
    * called want_new_episode().
    * responded to send_action(), and now we got the resulting next state.
    """
    def receive_new_state(self, state: tuple):
        self.states.append([(state[utils.IND_X], state[utils.IND_Y], state[utils.IND_Z])])

        is_state_too_bad = False #e.g. too far from other racing lines or too much time has passed <=> self.states array length is too big.
        if len(self.states) * utils.GAP_TIME >= utils.MAX_TIME:
            is_state_too_bad = True

        if is_state_too_bad:
            print(f"Episode too bad.")
            self.want_new_episode()


    """
    Called by the client to let us know that the episode ended by finishing the map.
    """
    def episode_ended(self):
        # ! in-time game is actually (len(self.states) - 1) * utils.GAP_TIME.
        utils.write_processed_output(
            fname = f"{utils.PROCESSED_OUTPUT_DIR_PREFIX}{str(time.time()).replace('.', '')}_{(len(self.states) - 1) * utils.GAP_TIME}.txt",
            actions = self.actions
        )

        self.want_new_episode()


    """
    Called by the client to ask for a new action.
    Will add a tuple (steer, gas, brake) to the internal self.actions list.
    The caller will access our internal self.actions list for further use.
    """
    def next_action(self):
        i = random.randint(0, 9)
        steer = utils.VAL_STEER_LEFT if i < 1 else (utils.VAL_NO_STEER if i < 9 else utils.VAL_STEER_RIGHT)

        gas = utils.VAL_GAS if random.randint(0, 99) < 99 else utils.VAL_NO_GAS
        brake = utils.VAL_NO_BRAKE if random.randint(0, 99) < 99 else utils.VAL_BRAKE

        self.actions[utils.IND_STEER].append(steer)
        self.actions[utils.IND_GAS].append(gas)
        self.actions[utils.IND_BRAKE].append(brake)
