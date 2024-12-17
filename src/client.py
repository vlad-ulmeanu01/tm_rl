from tminterface.interface import TMInterface
from tminterface.structs import Event
from tminterface.client import Client
import operator
import typing
import signal
import copy
import time
import sys

import agent_qnet
import utils


class MainClient(Client):
    def __init__(self):
        super().__init__()

        self.agent = agent_qnet.Agent()
        self.remembered_state = None

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
        if self.remembered_state is None and curr_time == -utils.GAP_TIME:
            self.remembered_state = iface.get_simulation_state()

        if curr_time >= 0:
            state = iface.get_simulation_state()

            x, y, z = [round(a, 5) for a in state.position]
            yaw, pitch, roll = [round(a, 5) for a in state.yaw_pitch_roll]
            vx, vy, vz = [round(a, 5) for a in state.velocity]
            wheel_materials = [state.simulation_wheels[i].real_time_state.contact_material_id for i in range(4)]
            wheel_has_contact = [1 if state.simulation_wheels[i].real_time_state.has_ground_contact else 0 for i in range(4)]

            self.agent.receive_new_state((x, y, z, yaw, pitch, roll, vx, vy, vz, wheel_materials, wheel_has_contact))

            if self.agent.agent_wants_new_episode: # after analyzing the newest state, the agent doesn't want to continue this episode anymore.
                self.on_checkpoint_count_changed(iface, current = -1, target = -1)
            else:
                self.agent.next_action() # knowing the new state, ask the agent to compute the next action. it adds it to self.agent.actions.

                # we change the event buffer data to account for the new action.
                iface.set_input_state(
                    sim_clear_buffer = True,
                    steer = self.agent.actions[utils.IND_STEER][-1],
                    accelerate = self.agent.actions[utils.IND_GAS][-1],
                    brake = self.agent.actions[utils.IND_BRAKE][-1]
                )


    def on_simulation_end(self, iface: TMInterface, result: int):
        print("This isn't supposed to happen.")


    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current < target:
            return

        iface.prevent_simulation_finish()

        t1 = time.time()
        self.agent.episode_ended(did_episode_end_normally = (target != -1))
        print(f"Expensive function took {round(time.time() - t1, 3)}s.")

        self.agent.agent_wants_new_episode = False

        iface.rewind_to_state(self.remembered_state)  # we restart the simulation.


    def on_laps_count_changed(self, iface, current: int):
        pass


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
