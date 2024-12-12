from tminterface.interface import TMInterface
import copy

import oracle

class Worker():
    def __init__(self):
        #comunicare cu jocul:
        self.should_client_work = False #variabila care indica DACA AR trebui sa ruleze clientul (ie e ceva in stiva)
        self.is_client_redoing = True #variabila care indica ralanti pentru simulari
        self.input_stack = [] #stiva cu [(input ce se doreste a fi rulat, [vector de stari rezultate din simulare: (x, y, z, y, p, r, vx, vy, vz, wh_mat, wh_contact])].
        self.input_stack_index = -1 #indica care input trebuie rulat acum.

        #starea initiala din care pornesc jocul si peste care adaug permanent.
        self.first_push = True #prima data cand bag in stiva nu trebuie sa analizez nimic.
        self.input_array = ([0], [1], [0]) #steer, gas, brake.

        #comunicare cu oracolul.
        self.oracle = oracle.Oracle()

        #deocamdata sunt multumit cu orice care termina traseul.
        self.tmp_did_finish_track = False

    def add_input_array_to_stack(self, input):
        self.input_stack.append([copy.deepcopy(input), []])
        self.input_stack_index += 1
        if not self.should_client_work:
            self.should_client_work = True

    #este apelat de client la sfarsitul unei simulari.
    def process_input_stack(self, iface: TMInterface):
        self.oracle.update_state_series(self.input_stack[self.input_stack_index][1])

        self.input_stack_index -= 1
        if self.input_stack_index < 0:
            self.should_client_work = False
            self.is_client_redoing = True

    def clear_stack(self):
        self.input_stack = []
        self.input_stack_index = -1

    def main_loop(self):
        if self.tmp_did_finish_track:
            return

        if not self.should_client_work:
            if len(self.input_stack) == 0:
                #trebuie dat de munca clientului.
                #trebuie bagate toate inputurile noi in acelasi timp in self.input_stack.

                if self.first_push: #este prima data cand bag in stiva, nu am nimic de analizat.
                    self.add_input_array_to_stack(self.input_array)
                    self.first_push = False
                else: #trebuie sa analizez ce mi-a dat jocul, sa adaug la input_array si sa bag din nou.
                    newSteer, newGas, newBrake = self.oracle.predict()
                    self.input_array[0].append(newSteer)
                    self.input_array[1].append(newGas)
                    self.input_array[2].append(newBrake)

                    self.add_input_array_to_stack(self.input_array)
            else:
                #daca sunt aici inseamna ca clientul a terminat munca pe care i-am dat-o, trebuie analizata si trebuie golita stiva.
                self.clear_stack()
        else:
            #lasa clientul sa lucreze ce i-ai dat.
            pass

        pass