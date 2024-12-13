from tminterface.structs import Event
from tminterface.interface import TMInterface
from tminterface.client import Client
import operator
import signal
import copy
import time
import sys

import worker
import sim_utils

class MainClient(Client):
    def __init__(self, w: worker.Worker):
        self.GAP_TIME = 10 #ms, lungimea unui cadru.
        self.IND_STEER, self.IND_GAS, self.IND_BRAKE = 0, 1, 2
        self.CUTOFF_TIME = 0

        self.processed_output_dir = "./processed_outputs/output_"

        self.last_time_in_sim_step = 0
        self.remembered_state = None
        self.did_race_finish = False

        self.worker = w
        #acest client este un fel de cutie neagra care trebuie sa primeasca niste inputuri, sa le ruleze
        #si sa returneze scorurile pentru inputuri
        #cand nu are nimic de rulat (i.e. is_client_redoing == True), clientul nu face nimic
        #cand primeste >= 1 inputuri, da reset la cursa, calculeaza pana le face pe toate etc.

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
        self.did_race_finish = False

    def on_simulation_step(self, iface: TMInterface, time: int):
        self.last_time_in_sim_step = time # - 2610

        if self.remembered_state == None and self.last_time_in_sim_step == -self.GAP_TIME:
            self.remembered_state = iface.get_simulation_state()

        if self.last_time_in_sim_step >= self.CUTOFF_TIME:
            self.on_checkpoint_count_changed(iface, -1, -1)
            #trebuie sa fortez terminarea cursei.. jocul se asteapta sa valideze ceva ce a terminat cursa

        if self.did_race_finish:
            #intotdeauna repet ultimul EventBufferData pana cand este modificat in on_checkpoint_count_changed
            #daca am ajuns aici sigur am trecut prin on_checkpoint_count_changed si am facut modificarile
            #la buffer daca trebuiau facute.
            iface.rewind_to_state(self.remembered_state)
            self.did_race_finish = False
        elif not self.worker.is_client_redoing:
            if time >= 0:
                state = iface.get_simulation_state()

                # adaug in input_stack informatii despre cadrul curent: x, y, z, y, p, r, vx, vy, vz, wh_mat, wh_contact.
                x, y, z = [round(a, 5) for a in state.position]
                yaw, pitch, roll = [round(a, 5) for a in state.yaw_pitch_roll]
                vx, vy, vz = [round(a, 5) for a in state.velocity]
                wheel_materials = [state.simulation_wheels[i].real_time_state.contact_material_id for i in range(4)]
                wheel_has_contact = [1 if state.simulation_wheels[i].real_time_state.has_ground_contact else 0 for i in
                                     range(4)]

                self.worker.input_stack[self.worker.input_stack_index][1].append(
                    (x, y, z, yaw, pitch, roll, vx, vy, vz, wheel_materials, wheel_has_contact))

    def on_simulation_end(self, iface: TMInterface, result: int):
        print("All simulations finished?? You weren't supposed to see this you know")

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current < target:
            return

        self.did_race_finish = True
        iface.prevent_simulation_finish()

        if not self.worker.is_client_redoing:
            if target > 0: #chiar am terminat traseul pe bune.
                self.worker.tmp_did_finish_track = True
                print(f"did finish track!")

                # ! acum scriu doar daca am terminat traseul. daca nu vrei, muta in afara if-ului.
                outName = f"{self.processed_output_dir}{str(time.time()).replace('.', '')}_{self.last_time_in_sim_step}.txt"
                sim_utils.write_processed_output(outName, self.worker.input_stack[self.worker.input_stack_index][0], self.GAP_TIME)
                print(f"wrote to {outName}!")

            self.worker.process_input_stack(iface)

        #daca nu am intrat in iful de mai sus, nu am avut ce sa rulez pentru un timp, asa ca am rulat ultima chestie din nou.
        #(self.worker.is_client_redoing, self.worker.should_client_work) in ((False, True), (True, False), (True, True))

        #self.worker.is_client_redoing, self.should_client_work sunt actualizate in self.process_input_stack mai sus daca e cazul
        if self.worker.should_client_work:
            #inseamna ca (optional mai) am ceva in stiva
            self.worker.is_client_redoing = False
            self.write_input_array_to_EventBufferData(iface, self.worker.input_stack[self.worker.input_stack_index][0])

    def on_laps_count_changed(self, iface, current: int):
        pass

    def write_input_array_to_EventBufferData(self, iface: TMInterface, input_array: list):
        assert(len(input_array[0]) == len(input_array[1]) and len(input_array[1]) == len(input_array[2]))
        n = len(input_array[0])
        self.CUTOFF_TIME = n * self.GAP_TIME #pun sa se termine fix dupa simularea ultimei bucati din input_array.

        ebd = copy.deepcopy(iface.get_event_buffer())
        ebd.events_duration = self.CUTOFF_TIME

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
        ebd.events.append(make_event(0, self.CUTOFF_TIME, "binary", 1)) #"buffer_end"

        #steer, push_up, push_down events
        for arr, event_type, value_type in ((input_array[self.IND_STEER], rev_control_names["Steer (analog)"], "analog"), #"Steer" nu mai exista.
                                            (input_array[self.IND_GAS], rev_control_names["Accelerate"], "binary"),
                                            (input_array[self.IND_BRAKE], rev_control_names["Brake"], "binary")):
            ebd.events.append(make_event(event_type, 0, value_type, arr[0]))
            for i in range(1, len(arr)):
                if arr[i] != arr[i-1]:
                    ebd.events.append(make_event(event_type, i * self.GAP_TIME, value_type, arr[i]))

        ebd.events.sort(key = operator.attrgetter("time"), reverse = True)
        iface.set_event_buffer(ebd)

def main():
    server_name = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server_name}...")

    client = MainClient(worker.Worker())
    iface = TMInterface(server_name, 65536)

    def handler(signum, frame):
        iface.close()

    if sys.platform == 'win32':
        signal.signal(signal.SIGBREAK, handler)
    signal.signal(signal.SIGINT, handler)

    iface.register(client)
    while iface.running:
        client.worker.main_loop()
        time.sleep(0)

if __name__ == '__main__':
    # main client -> worker -> oracle
    #                                  -> worker -> main client.
    main()