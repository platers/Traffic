#!/usr/bin/env python
"""
@file    runner.py
@author  Lena Kalleske
@author  Daniel Krajzewicz
@author  Michael Behrisch
@author  Jakob Erdmann
@date    2009-03-26
@version $Id: runner.py 20433 2016-04-13 08:00:14Z behrisch $

Tutorial for traffic light control via the TraCI interface.

SUMO, Simulation of Urban MObility; see http://sumo.dlr.de/
Copyright (C) 2009-2016 DLR/TS, Germany

This file is part of SUMO.
SUMO is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random
import time
import numpy as np
from qlearning import DeepQ
from qlearning import ExperienceReplay

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

import traci
# the port used for communicating with your sumo instance
PORT = 8873

startTime = {}

def generate_routefile():
    random.seed(42)  # make tests reproducible
    N = 3600000  # number of time steps
    # demand per second from different directions
    with open("data/cross.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>""", file=routes)
        f = 0
        for ep in range(500):
            b = ep * 500

            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    print('    <flow id="%i" begin="%i" end="%i" probability="%f" type="typeWE" from="%ii" to="%io"/>' % (f, b, b + 100, 0.005, i, j), file=routes)
                    f += 1
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue        
                    print('    <flow id="%i" begin="%i" end="%i" probability="%f" type="typeWE" from="%ii" to="%io"/>' % (f, b + 100, b + 200, 0.007, i, j), file=routes)
                    f += 1
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    print('    <flow id="%i" begin="%i" end="%i" probability="%f" type="typeWE" from="%ii" to="%io"/>' % (f, b + 200, b + 300, 0.009, i, j), file=routes)
                    f += 1
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    print('    <flow id="%i" begin="%i" end="%i" probability="%f" type="typeWE" from="%ii" to="%io"/>' % (f, b + 300, b + 400, 0.007, i, j), file=routes)
                    f += 1
            for i in range(8):
                for j in range(8):
                    if i == j:
                        continue
                    print('    <flow id="%i" begin="%i" end="%i" probability="%f" type="typeWE" from="%ii" to="%io"/>' % (f, b + 400, b + 500, 0.0045, i, j), file=routes)
                    f += 1
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

def actionState(l):
    actions = []
    if traci.trafficlights.getPhase(l) % 2 == 0:
        # we are not already switching
        actions.append(0)
        actions.append(1)
    return actions

def setState(light, action):
    #0 is default state, 1 is other state
    s = traci.trafficlights.getPhase(light)
    if action == 0:
        if s == 0:
            traci.trafficlights.setPhase(light, 0)
        else:
            traci.trafficlights.setPhase(light, 3)
            return 1
    if action == 1:
        if s == 2:
            traci.trafficlights.setPhase(light, 2)
        else:
            traci.trafficlights.setPhase(light, 1)
            return 1
    return 0

e = 1

def getReward(changed):
    lanes = traci.lane.getIDList()
    s = 0.0
    for l in lanes:
        q, w, speed = 0, 0, 0
        for c in traci.lane.getLastStepVehicleIDs(l):
            if traci.vehicle.getWaitingTime(c) > 0:
                q += 1
            speed += traci.vehicle.getSpeed(c)
        speed /= max(1, len(traci.lane.getLastStepVehicleIDs(l))) * 10
        w = traci.lane.getWaitingTime(l)
        #s -= 1 * (q) ** 1.5 + 2 * (w) ** 1.5 - speed ** 1.5
        s += speed
    #s /= max(len(traci.vehicle.getIDList()), 1)
    for l in lanes:        
        s -= traci.lane.getCO2Emission(l) / 10000.0
    #s -= changed
    return s

def getCO2():
    s = 0
    for l in traci.lane.getIDList():
        s += traci.lane.getCO2Emission(l)
    return s

def getWaitingTime():
    s = 0
    for l in traci.lane.getIDList():
        for c in traci.lane.getLastStepVehicleIDs(l):
            #if traci.vehicle.getWaitingTime(c) > 0:
                #s += 1
            s += traci.vehicle.getWaitingTime(c)
    return s


def step(lights, loops, history, cars_detected, actions):
    light_state = [0] * len(lights)
    cars_total = [0] * len(loops)
    changed = 0
    for l in lights:
        changed += setState(l, actions[int(l)])
    traci.simulationStep()
    for l in lights:
        light_state[int(l)] = traci.trafficlights.getPhase(l)
        history[int(l)].append(traci.trafficlights.getPhase(l))
        history[int(l)].pop(0)
    for i in loops:
        cars_detected[int(i)].append(traci.inductionloop.getLastStepVehicleNumber(i))
        cars_detected[int(i)].pop(0)
        cars_total[int(i)] = sum(cars_detected[int(i)])
    ss = light_state
    ss = np.append(ss, sum(history, []))
    cars = traci.vehicle.getIDList()
    r = getReward(changed)
    #for c in cars:
        #travel_time = traci.simulation.getCurrentTime() / 1000 - startTime[c]
        #r -= travel_time
    return ss, r, cars_detected, history


def run():
    """execute the TraCI control loop"""
    # first, generate the route file for this simulation
    generate_routefile()
    traci.init(PORT)
    loops = traci.inductionloop.getIDList()
    lights = traci.trafficlights.getIDList()
    history_len = 5
    input_size = len(lights) + history_len * len(lights)
    num_actions = 2
    update_target = 10
    batch_size = 32
    e = 1
    exp_replay = [ExperienceReplay()] * len(lights)
    DQN = [DeepQ(input_size, num_actions)] * len(lights)
    maxSteps = 500
    total_steps = 0
    totalCO2, totalWaitingTime = [], []
    for iteration in xrange(0, 1000):
        R, t, = 0, 0
        cars_detected = [[0 for i in range(20)] for j in range(len(loops))]
        history = [[0 for i in range(history_len)] for j in range(len(lights))]
        s = [0] * (input_size)
        actions = [0] * len(lights)
        totalCO2.append(0)
        totalWaitingTime.append(0)
        for l in lights:
            traci.trafficlights.setPhase(l, 0)
        while t < maxSteps:
            for l in lights:
                a = -1
                if len(actionState(l)) == 0 or t % 15 != 0:
                    actions[int(l)] = -1
                    continue
                elif np.random.uniform() < e:
                    a = np.random.choice(actionState(l)) 
                else:
                    a = DQN[int(l)].getAction(s)
                actions[int(l)] = a
            ss, r, cars_detected, history = step(lights, loops, history, cars_detected, actions)
            R += r
            totalWaitingTime[len(totalWaitingTime) - 1] += getWaitingTime()
            totalCO2[len(totalWaitingTime) - 1] += getCO2()
            for l in lights:
                if actions[int(l)] == -1:
                    continue
                exp_replay[int(l)].remember([s, actions[int(l)], r, ss], False)
                if total_steps % update_target == 0:
                    DQN[int(l)].updateTarget()
                DQN[int(l)].trainModel(exp_replay[int(l)].get_batch(batch_size), 0.9, input_size, num_actions)
            t += 1
            total_steps += 1
            if e > 0.01:
                e *= 0.9998
            s = ss
        totalWaitingTime[len(totalWaitingTime) - 1] /= t
        totalCO2[len(totalWaitingTime) - 1] /= t
        if iteration > 10:
            totalWaitingTime.pop(0)
            totalCO2.pop(0)
        print('Iteration %i completed with Average CO2: %i and Average waiting time %i reward %i epsilon %f' % (iteration, sum(totalCO2) / len(totalCO2), sum(totalWaitingTime) / len(totalWaitingTime), R, e))
        for car in traci.vehicle.getIDList():
            traci.vehicle.remove(car)
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    sumoProcess = subprocess.Popen([sumoBinary, "-c", "data/cross.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml", "--remote-port", str(PORT)], stdout=sys.stdout, stderr=sys.stderr)
    run()
    sumoProcess.wait()
