import pandas as pd
import numpy as np
import sklearn
import random
import torch
import copy
import sys

import sim_utils

"""
oracolul primeste input din simulare si decide care este urmatoarea actiune.
"""
class Oracle:
    def __init__(self):
        self.state_series = [] #[(x (0), y (1), z (2), yaw (3), pitch (4), roll (5), vx (6), vy(7), vz (8), wh_mat (list, 9), wh_contact (list, 10))].
        self.IND_X, self.IND_Y, self.IND_Z = 0, 1, 2
        self.IND_YAW, self.IND_PITCH, self.IND_ROLL, self.IND_VX, self.IND_VY, self.IND_VZ, self.IND_WHEEL_MATERIALS, self.IND_WHEEL_CONTACT = 3, 4, 5, 6, 7, 8, 9, 10

        # TODO oracle ar tb sa mai aiba o functie care sa il anunte ca a terminat replay-ul ai sa isi updateze parametrii interni.

    """
    apelat de worker cand termina o simulare, primesc de la el starile prin care a trecut de-a lungul simularii.
    (old: primesc informatie noua de la worker).
    """
    def update_state_series(self, state_series: list):
        print(f"(oracle) new length: {len(state_series)}.")
        #print(state_series)
        self.state_series = copy.deepcopy(state_series)

        if len(self.state_series) > 0: # (?) reset informatii material pentru prima stare.
            self.state_series[0] = (*self.state_series[0][:-2], [16] * 4, [1] * 4)

    """
    tinand cont de state_series, prezice urmatorul input, un tuplu (steer, gas, brake).
    """
    def predict(self):
        # TODO aici trebuie sa fie implementarea.
        gasValue = 1
        brakeValue = 0
        # steerValue = random.choice([-65536, 0, 65536])
        steerValue = random.choice([-65536, 0, 0, 0, 0, 0, 0, 0, 65536])

        #print(f"predicted: steerValue = {steerValue}, gasValue = {gasValue}, brakeValue = {brakeValue}.")

        return steerValue, gasValue, brakeValue
