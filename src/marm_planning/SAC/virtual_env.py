#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.insert(0, "/home/you/robotiq/src/hrl-kdl-indigo-devel/hrl_geom/src")
sys.path.insert(0, "/home/you/robotiq/src/Basic_math/hrl-kdl-indigo-devel/pykdl_utils/src")

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics


class abb_env:
    def __init__(self):
        self.robot = URDF.from_xml_file("/home/you/abb/590.urdf")
        self.tree = kdl_tree_from_urdf_model(self.robot)
        # print self.tree.getNrOfSegments()
        self.chain = self.tree.getChain("base", "tool0")
        # print self.chain.getNrOfJoints()
        self.kdl_kin = KDLKinematics(self.robot, "base_link", "link_6", self.tree)
        self.steps = 0
        self.done = False
        self.target_position = [0.5] * 3
        self.r = 0.0
        self.success_dist = 0.01
        self.success_dist_min = 0.01
        self.link = 'panda_hand'
        self.distance = [0.0] * 3
        self.random_pos = []
        self.a_bound = [2.9, 1.88, 2.2]
        self.offset = [0.0, 0.20, -1.1]
        # self.a_bound = [0.0, 1.88, 2.2]
        # self.offset = [0.0, 0.20, -1.1]
        self.reference_frame = 'base_link'

    def get_random_pos(self):
        a = np.random.uniform(-1, 1, 3) * self.a_bound
        a += self.offset
        a = self.action_limit(a)
        # print a
        pose1 = np.array(self.kdl_kin.forward(a))  # forward kinematics (returns homogeneous 4x4 numpy.mat)
        pose = pose1[:-1, 3]
        self.random_pos.append(pose)

    def set_target_position(self):
        np.random.shuffle(self.random_pos)
        pos = self.random_pos[0][:]
        self.target_position = pos
        # print "target has updated to:" + str(pos)

    def action(self, action):
        action = self.action_limit(action)
        pose1 = np.array(self.kdl_kin.forward(action))
        pose = pose1[:-1, 3]
        self.distance = pose - self.target_position

    def reset(self, reset_action):
        self.steps = 0
        self.done = False
        reset_action = reset_action[:]
        state = []
        state.extend(reset_action)
        self.action(reset_action)
        state.extend(np.array(self.target_position))
        return state

    def step(self, action):
        self.steps += 1
        state = []
        state.extend(action)
        self.action(action)
        state.extend(self.target_position)
        dist_new = pow(pow(self.distance[0], 2) + pow(self.distance[1], 2) + pow(self.distance[2], 2), .5)
        self.r = (np.exp(-0.99 * dist_new) - 1) * 2
        # self.r = -10*dist_new
        if dist_new <= self.success_dist:
            self.done = True
        if self.done:
            self.steps = 0
        return state, self.r, self.done, dist_new

    def target_dist_update(self):
        if self.success_dist > self.success_dist_min:
            self.success_dist *= 0.8
            self.success_dist = max(self.success_dist, self.success_dist_min)
            print '*' * 20
            print 'success_dist has updated to:' + str(self.success_dist)
            print '*' * 20

    def action_limit(self, action):
        a_bound = self.a_bound
        offset = self.offset
        a = action[:]
        for c in range(3):
            while a[c] > a_bound[c] + offset[c]:
                a[c] = a_bound[c] + offset[c]
            while a[c] < -a_bound[c] + offset[c]:
                a[c] = -a_bound[c] + offset[c]
        if a[1] > 0.45 and a[2] > 1.15:
            a[2] = np.random.uniform(1.0, 1.15)
        if a[1] + a[2] > 1.85 and a[2] > 0:
            a[2] = np.random.uniform(0.9 * (1.85 - a[1]), 1.85 - a[1])
        if type(a) == np.ndarray:
            a = a.tolist()
        a.extend([0.0] * 3)
        return a


if "__name__" == "__main__":
    enV = abb_env

    # q = [0,0,0,0,0,0]
    # pose = kdl_kin.forward(q) # forward kinematics (returns homogeneous 4x4 numpy.mat)
    # q_ik = kdl_kin.inverse(pose) # inverse kinematics
    # if q_ik is not None:
    #     pose_sol = kdl_kin.forward(q_ik) # should equal pose
    # #J = kdl_kin.jacobian(q)
    # print 'q:', q
    # print 'pose:', pose
    # print 'q_ik:', q_ik
    # if q_ik is not None:
    #     print 'pose_sol:', pose_sol
