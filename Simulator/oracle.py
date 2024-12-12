import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.ioff()

import pandas as pd
import numpy as np
import sklearn
import torch
import copy

import sim_utils

import sys

sys.path.append("C:/Users/ulmea/Documents/GitHub/tm_nn/Network")
import classes
sys.path.append("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeRefined")
import refine_utils
sys.path.append("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeUnrefined")
import nr_utils
import conv_make_input_from_csv_pair

"""
oracolul primeste input din simulare si decide care este urmatoarea actiune.
"""
class Oracle:
    def __init__(self):
        self.state_series = [] #[(x (0), y (1), z (2), yaw (3), pitch (4), roll (5), vx (6), vy(7), vz (8), wh_mat (list, 9), wh_contact (list, 10))].
        self.IND_X, self.IND_Y, self.IND_Z = 0, 1, 2
        self.IND_YAW, self.IND_PITCH, self.IND_ROLL = 3, 4, 5
        self.IND_VX, self.IND_VY, self.IND_VZ = 6, 7, 8
        self.IND_WHEEL_MATERIALS = 9
        self.IND_WHEEL_CONTACT = 10

        self.ref = refine_utils.Refiner("C:/Users/ulmea/Documents/GitHub/tm_nn/MakeRefined/export_pts_conv_lg_raport.txt")

        self.dfrRacingLine = pd.read_csv("racing_line_TMN_A-0.csv", skipinitialspace = True)

        self.netSteer = classes.MainNet(3)
        self.netSteer.load_state_dict(torch.load("NetTM_best_steer_steady_5_12gen.pt")) #NetTM_best_BatchNorm
        self.netSteer.eval()

        # self.netBrake = classes.MainNet(2)
        # self.netBrake.load_state_dict(torch.load("NetTM_best_brake_steady5_5gen.pt"))
        # self.netBrake.eval()

        #de cat timp franez, cand am franat ultima data.
        self.timeSinceLastBrake, self.timeSpentBraking = 0, 0
        self.timeSinceLastAir, self.timeSpentAir = 0, 0

        #mentin local o copie a inputului.
        self.GAP_TIME = 10 #ms, lungimea unui cadru.
        self.IND_STEER, self.IND_GAS, self.IND_BRAKE = 0, 1, 2
        self.localInput = [[0], [1], [0]]

        #lucruri de la unrefined_input.
        self.n = [0, len(self.dfrRacingLine["time"])]
        self.time = [[], nr_utils.normalize([self.dfrRacingLine["time"][i] for i in range(self.n[1])], m=0, M=nr_utils.MAX_VALUE_TIME)]
        self.xs = [[], nr_utils.normalize([self.dfrRacingLine["x"][i] for i in range(self.n[1])], m=0, M=nr_utils.MAX_VALUE_XZ)]
        self.ys = [[], nr_utils.normalize([self.dfrRacingLine["y"][i] for i in range(self.n[1])], m=0, M=nr_utils.MAX_VALUE_Y)]
        self.zs = [[], nr_utils.normalize([self.dfrRacingLine["z"][i] for i in range(self.n[1])], m=0, M=nr_utils.MAX_VALUE_XZ)]

        self.kdt = sklearn.neighbors.KDTree([[self.xs[1][i], self.ys[1][i], self.zs[1][i]] for i in range(self.n[1])], leaf_size=30, metric="euclidean")



    """
    primesc informatie noua de la worker.
    """
    def update_state_series(self, state_series: list):
        print(f"(oracle) new length: {len(state_series)}.")
        #print(state_series)
        self.state_series = copy.deepcopy(state_series)

        if len(self.state_series) > 0:
            self.state_series[0] = (*self.state_series[0][:-2], [16] * 4, [1] * 4)

    """
    tinand cont de state_series, prezice urmatorul input, tuplu (steer, gas, brake).
    """
    def predict(self):
        netInput = [] #construiesc gradual ce bag in retea.

        refSpeed = refine_utils.refineSpeed(np.linalg.norm(np.array(
            [self.state_series[-1][self.IND_VX], self.state_series[-1][self.IND_VY], self.state_series[-1][self.IND_VZ]])) * 3.6)
        netInput.extend(refSpeed)

        if self.localInput[self.IND_BRAKE][-1] == 1:
            self.timeSinceLastBrake = 0
            self.timeSpentBraking += self.GAP_TIME
        else:
            self.timeSinceLastBrake += self.GAP_TIME
            self.timeSpentBraking = 0

        material = [0] * 4
        modeMaterial = nr_utils.getMode([self.state_series[-1][self.IND_WHEEL_MATERIALS][j] for j in range(4)], 32)
        if sum([self.state_series[-1][self.IND_WHEEL_CONTACT][j] == 0 for j in range(4)]) >= 2:  #consider ca sunt in aer.
            material = [0, 1, 0, 0]
            self.timeSinceLastAir = 0
            self.timeSpentAir += self.GAP_TIME
        else:
            self.timeSinceLastAir += 10
            self.timeSpentAir = 0
            if modeMaterial == 2:  #grass.
                material = [0, 0, 0, 1]
            elif modeMaterial == 6:  #dirt.
                material = [0, 0, 1, 0]
            else: #road.
                material = [1, 0, 0, 0]
        netInput.extend(material)

        netInput.extend(self.ref.refineValue("timeSinceLastBrake", self.timeSinceLastBrake))
        netInput.extend(self.ref.refineValue("timeSpentBraking", self.timeSpentBraking))
        netInput.extend(self.ref.refineValue("timeSinceLastAir", self.timeSinceLastAir))
        netInput.extend(self.ref.refineValue("timeSpentAir", self.timeSpentAir))

        #TODO optimizare si aici la cat bagi in moveOriginTo.
        self.n[0] = len(self.state_series)
        self.time[0] = nr_utils.normalize([i * self.GAP_TIME for i in range(self.n[0])], m = 0, M = nr_utils.MAX_VALUE_TIME)
        self.xs[0] = nr_utils.normalize([self.state_series[i][self.IND_X] for i in range(self.n[0])], m = 0, M = nr_utils.MAX_VALUE_XZ)
        self.ys[0] = nr_utils.normalize([self.state_series[i][self.IND_Y] for i in range(self.n[0])], m = 0, M = nr_utils.MAX_VALUE_Y)
        self.zs[0] = nr_utils.normalize([self.state_series[i][self.IND_Z] for i in range(self.n[0])], m = 0, M = nr_utils.MAX_VALUE_XZ)

        l = self.n[0] - 1
        pre_l = max(0, l - nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH + 1)
        indexClosestPtR2 = int(self.kdt.query([[self.xs[0][l], self.ys[0][l], self.zs[0][l]]], k=1, return_distance=False))

        # rotesc sistemul de coordonate ai (xs[0][l], ys[0][l], zs[0][l]) sa fie originea.
        tmpXs, tmpYs, tmpZs = [None] * 2, [None] * 2, [None] * 2

        tmpXs[0], tmpYs[0], tmpZs[0] = conv_make_input_from_csv_pair.moveOriginTo(
            (self.xs[0][l], self.ys[0][l], self.zs[0][l],
             self.state_series[l][self.IND_YAW], self.state_series[l][self.IND_PITCH], self.state_series[l][self.IND_ROLL]),
            self.n[0], self.xs[0], self.ys[0], self.zs[0])
        tmpXs[1], tmpYs[1], tmpZs[1] = conv_make_input_from_csv_pair.moveOriginTo(
            (self.xs[0][l], self.ys[0][l], self.zs[0][l],
             self.state_series[l][self.IND_YAW], self.state_series[l][self.IND_PITCH], self.state_series[l][self.IND_ROLL]),
            self.n[1], self.xs[1], self.ys[1], self.zs[1])

        #ma asigur ca am fix 150 de puncte din trecut si 150 de puncte din viitor.
        tryXs = nr_utils.padLR(tmpXs[0][pre_l: l + 1], nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH, "l") + \
                nr_utils.padLR(tmpXs[1][indexClosestPtR2: indexClosestPtR2 + nr_utils.MIN_INTERVAL_LENGTH],
                               nr_utils.MIN_INTERVAL_LENGTH, "r")
        tryYs = nr_utils.padLR(tmpYs[0][pre_l: l + 1], nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH, "l") + \
                nr_utils.padLR(tmpYs[1][indexClosestPtR2: indexClosestPtR2 + nr_utils.MIN_INTERVAL_LENGTH],
                               nr_utils.MIN_INTERVAL_LENGTH, "r")
        tryZs = nr_utils.padLR(tmpZs[0][pre_l: l + 1], nr_utils.MAIN_REPLAY_PRECEDENT_LENGTH, "l") + \
                nr_utils.padLR(tmpZs[1][indexClosestPtR2: indexClosestPtR2 + nr_utils.MIN_INTERVAL_LENGTH],
                               nr_utils.MIN_INTERVAL_LENGTH, "r")

        tryXs = conv_make_input_from_csv_pair.timeSeriesModify(tryXs)
        tryYs = conv_make_input_from_csv_pair.timeSeriesModify(tryYs)
        tryZs = conv_make_input_from_csv_pair.timeSeriesModify(tryZs)

        # plt.plot(tryZs)
        # plt.savefig(f"C:/Users/ulmea/Desktop/Probleme/Trackmania/test_simulator/dbg_turn_left/{len(self.state_series)}.png")
        # plt.close()

        for i in range(97):
            netInput.extend(self.ref.refineValue('x', tryXs[i]))
        for i in range(97):
            netInput.extend(self.ref.refineValue('y', tryYs[i]))
        for i in range(97):
            netInput.extend(self.ref.refineValue('z', tryZs[i]))
        # netInput.extend(self.ref.refineValue('x', tryXs[80]))
        # print(netInput)

        #print(f"netInput size = {len(netInput)}.")
        yPredSteer = self.netSteer(torch.FloatTensor(netInput).unsqueeze(0)) #acum orice tuplu din yPred e de forma 1, *; mai tb un [0] in mijloc la adresare.
        #yPredBrake = self.netBrake(torch.FloatTensor(netInput).unsqueeze(0))

        # print(f"steer: {[round(x.item(), 3) for x in yPredSteer[0]]}, brake: {[round(x.item(), 3) for x in yPredBrake[0]]}.")
        print(f"steer: {[round(x.item(), 3) for x in yPredSteer[0]]}.")

        gasValue = 1 #1 if yPred[0][0][0] > yPred[0][0][1] else 0
        brakeValue = 0 #1 if yPredBrake[0][0] > yPredBrake[0][1] else 0
        steerValue = refine_utils.reverseGetSimpleSteer([x.item() for x in yPredSteer[0]])

        #print(f"predicted: steerValue = {steerValue}, gasValue = {gasValue}, brakeValue = {brakeValue}.")

        return steerValue, gasValue, brakeValue
