from tminterface.interface import TMInterface
from tminterface.structs import Event
from tminterface.client import Client
import operator
import typing
import signal
import copy
import time
import sys

import agent_example
import utils


class MainClient(Client):
    def __init__(self):
        super().__init__()

        self.agent = agent_example.Agent()

        self.remembered_state = None
        self.tmp_states = [] # will keep all the states that EventBufferData goes through. Unfortunately, only the last one will be useful to the agent.

    def on_registered(self, iface: TMInterface):
        print(f"Registered to {iface.server_name}.")

    def on_deregistered(self, iface):
        pass

    def on_shutdown(self, iface):
        pass

    def on_run_step(self, iface: TMInterface, time: int):
        pass

    def on_simulation_begin(self, iface: TMInterface):
        iface.remove_state_validation()


    def on_simulation_step(self, iface: TMInterface, curr_time: int):
        if self.remembered_state is None and curr_time == -utils.GAP_TIME: # TODO fixeaza on_simulation_end dupa un run ok?
            self.remembered_state = iface.get_simulation_state()

        if curr_time >= 0:
            state = iface.get_simulation_state()

            x, y, z = [round(a, 5) for a in state.position]
            yaw, pitch, roll = [round(a, 5) for a in state.yaw_pitch_roll]
            vx, vy, vz = [round(a, 5) for a in state.velocity]
            wheel_materials = [state.simulation_wheels[i].real_time_state.contact_material_id for i in range(4)]
            wheel_has_contact = [1 if state.simulation_wheels[i].real_time_state.has_ground_contact else 0 for i in range(4)]

            self.tmp_states.append((x, y, z, yaw, pitch, roll, vx, vy, vz, wheel_materials, wheel_has_contact))

        if curr_time >= len(self.agent.states) * utils.GAP_TIME:
            # the event buffer data has no more actions to use from now on.
            # we forcefully finish the in-game race by calling on_checkpoint_count_changed with current = target = -1.
            self.on_checkpoint_count_changed(iface, current = -1, target = -1)


    def on_simulation_end(self, iface: TMInterface, result: int):
        print("This isn't supposed to happen.")


    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current < target:
            return

        iface.prevent_simulation_finish()

        if target == -1:
            # the run didn't actually finish, we just don't have enough in the event buffer data to go further.
            # we arrive in this branch from a call from on_simulation_step.

            self.agent.receive_new_state(self.tmp_states[-1]) # only the last tmp_state is new to the agent.

            if self.agent.agent_wants_new_episode: # after analyzing the newest state, the agent doesn't want to continue this episode anymore.
                self.agent.agent_wants_new_episode = False # we expect the agent to have switched agent_wants_new_episode through want_new_episode()
            else:
                self.agent.next_action() # knowing the new state, ask the agent to compute the next action. it adds it to self.agent.actions.
        else:
            self.agent.episode_ended() # the episode actually ended. notify the agent. we expect it to also call want_new_episode.
            self.agent.agent_wants_new_episode = False

        # we redo the whole simulation, either:
        # * adding the new action to the mix (we took the 'else' branch in target == -1).
        # * resetting the actions array and starting over with a new episode (we took the 'if' branch in target == -1 or the 'else' branch).
        iface.rewind_to_state(self.remembered_state)

        self.tmp_states = []
        self.write_input_array_to_EventBufferData(iface, self.agent.actions)


    def on_laps_count_changed(self, iface, current: int):
        pass


    def write_input_array_to_EventBufferData(self, iface: TMInterface, actions: typing.Tuple[typing.List, typing.List, typing.List]):
        assert(len(actions[utils.IND_STEER]) == len(actions[utils.IND_GAS]) and len(actions[utils.IND_GAS]) == len(actions[utils.IND_BRAKE]))
        n = len(actions[utils.IND_STEER])

        # we cut the simulation immediately after the last action.
        # 0 actions: keep until 0ms, we want to read the initial state.
        # 1 action: keep until 10ms to read the state after the only acton.
        cutoff_time = n * utils.GAP_TIME

        ebd = copy.deepcopy(iface.get_event_buffer())
        ebd.events_duration = cutoff_time

        rev_control_names = {ebd.control_names[i]: i for i in range(len(ebd.control_names))}

        def make_event(event_type: int, time: int, value_type: str, value: int) -> Event:
            ev = Event(100010 + time, 0)
            ev.name_index = event_type
            if value_type == "binary":
                ev.binary_value = value
            elif value_type == "analog":
                ev.analog_value = value
            else:
                print(f"(make_event) no such value_type as {value_type}!")
                assert(False)
            return ev

        ebd.events = []
        ebd.events.append(make_event(rev_control_names["_FakeIsRaceRunning"], -10, "binary", 1))
        ebd.events.append(make_event(0, cutoff_time, "binary", 1)) #"buffer_end"

        #steer, push_up, push_down events
        for arr, event_type, value_type in ((actions[utils.IND_STEER], rev_control_names["Steer (analog)"], "analog"), #"Steer" nu mai exista.
                                            (actions[utils.IND_GAS], rev_control_names["Accelerate"], "binary"),
                                            (actions[utils.IND_BRAKE], rev_control_names["Brake"], "binary")):
            if arr:
                ebd.events.append(make_event(event_type, 0, value_type, arr[0]))
                for i in range(1, len(arr)):
                    if arr[i] != arr[i-1]:
                        ebd.events.append(make_event(event_type, i * utils.GAP_TIME, value_type, arr[i]))

        ebd.events.sort(key = operator.attrgetter("time"), reverse = True)
        iface.set_event_buffer(ebd)


def run_client():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server_name}...")

    client = MainClient()
    iface = TMInterface(server_name, 65536)

    def handler(signum, frame):
        iface.close()

    if sys.platform == 'win32':
        signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    iface.register(client)
    while iface.running:
        time.sleep(0)


if __name__ == '__main__':
    run_client()
