#!/usr/bin/env python
# -*- coding: utf-8 -*-
import client1
import numpy as np


class Env:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.target_position = [0.0, 0.0, 0.0]
        self.r = 0.0
        self.target_dist = 0.5
        self.target_dist_min = 0.1
        self.finished_times = 0
        self.lalast_action = []

    def set_target_position(self, target_position):
        self.target_position = target_position

    def reset(self, reset_action):
        self.steps = 0
        self.done = False
        reset_action = reset_action[:]
        reset_action.extend([0.0, 0.0, 0.0])
        state = client1.client(str(reset_action))
        state = eval(state)
        state = state[:3]
        state.extend(self.target_position)
        return state

    def step(self, action):
        self.steps += 1
        action = action[:]
        action.extend([0.0, 0.0, 0.0])
        next_state = client1.client(str(action))
        next_state = eval(next_state)
        next_state = next_state[:3]
        next_state.extend(self.target_position)
        dist_new = pow(pow(next_state[0] - self.target_position[0], 2) + pow(next_state[1] - self.target_position[1], 2)
                       + pow(next_state[2] - self.target_position[2], 2), .5)
        self.r = 2 * (np.exp(-dist_new) - 1)
        if dist_new <= self.target_dist:
            self.done = True
            self.finished_times += 1
            self.r += 250
            if self.finished_times % 5 == 0 and self.finished_times != 0 and self.target_dist > self.target_dist_min:
                self.target_dist *= 0.8
        if self.steps >= 150:
            self.done = True
        if self.done:
            self.steps = 0

        if not self.done:
            if self.steps == 2:
                self.lalast_action = action[:]
            if self.steps % 2 == 0 and self.steps != 2:
                difference = np.tanh((action[0] - self.lalast_action[0]) ** 2 + (action[1] - self.lalast_action[1]) ** 2
                                     + (action[2] - self.lalast_action[2]) ** 2)
                self.r += np.tanh(difference - 0.1)
                self.lalast_action = action[:]
        return next_state, self.r, self.done, dist_new


if "__name__" == "__main__":
    enV = Env
