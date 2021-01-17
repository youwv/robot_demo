#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
use random start joint states
"""
import numpy as np
import rospy
import moveit_commander
import sys
import os
import cv2
from combination import get_picture


class Env:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.target_position = [0.5] * 3
        self.r = 0.0
        self.success_dist = 0.1
        self.success_dist_min = 0.1
        self.link = 'panda_hand'
        self.link_pos = [0.0] * 3
        self.random_pos = []
        self.a_bound = [2.9, 1.88, 2.2]
        self.offset = [0.0, 0.20, -1.1]
        self.count = 0
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        # 初始化ROS节点
        rospy.init_node('moveit_fk_demo', anonymous=True)
        # 初始化需要使用move group控制的机械臂中的arm group
        self.arm = moveit_commander.MoveGroupCommander('arm')
        # 设置目标位置所使用的参考坐标系
        self.reference_frame = 'base_link'
        self.arm.set_pose_reference_frame(self.reference_frame)
        os.system('rosrun gazebo_ros spawn_model '
                  '-file /home/you/robotiq/src/marm_planning/test_version/object.urdf -urdf -z 1 -model my_box')

    def get_random_pos(self):
        a = np.random.uniform(-1, 1, 3) * self.a_bound / 5
        a += self.offset
        a = self.action_limit(a)
        # print a
        self.arm.set_joint_value_target(a)
        self.arm.set_start_state_to_current_state()
        self.arm.go()
        pose = self.arm.get_current_pose(self.link)
        self.random_pos.append([round(pose.pose.position.x, 3),
                                round(pose.pose.position.y, 3), round(pose.pose.position.z, 3)])

    def set_target_position(self):
        np.random.shuffle(self.random_pos)
        pos = self.random_pos[0][:]
        self.target_position = pos
        os.system("rosservice call /gazebo/set_model_state '{model_state: { model_name: my_box, pose: "
                  "{ position: { x: %f, y: %f ,z: %f }} , reference_frame: world } }'"
                  % (float(pos[0]), float(pos[0]), float(pos[0])))
        print "target has updated to:" + str(pos)

    def action(self, action):
        action = self.action_limit(action)
        self.arm.set_joint_value_target(action)
        self.arm.set_start_state_to_current_state()
        self.arm.go()
        pose = self.arm.get_current_pose(self.link)
        self.link_pos[0] = round(pose.pose.position.x - self.target_position[0], 3)
        self.link_pos[1] = round(pose.pose.position.y - self.target_position[1], 3)
        self.link_pos[2] = round(pose.pose.position.z - self.target_position[2], 3)

    def reset(self, reset_action):
        self.steps = 0
        self.done = False
        reset_action = reset_action[:]
        self.action(reset_action)
        image = get_picture()
        return image

    def step(self, action):
        self.steps += 1
        self.count += 1
        self.action(action)
        image = get_picture()
        dist_new = pow(pow(self.link_pos[0], 2) + pow(self.link_pos[1], 2) + pow(self.link_pos[2], 2), .5)
        # display the distance every 20 steps
        if self.steps % 40 == 0:
            print 'dist:' + str(round(dist_new, 3))
        self.r = (np.exp(-0.99*dist_new)-1)*2
        if dist_new <= self.success_dist:
            self.done = True
        if self.done:
            self.steps = 0
        return image, self.r, self.done, dist_new

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
                a[c] = a_bound[c]+offset[c]
            while a[c] < -a_bound[c] + offset[c]:
                a[c] = -a_bound[c]+offset[c]
        if a[1] > 0.45 and a[2] > 1.15:
            a[2] = np.random.uniform(1.0, 1.15)
        if a[1] + a[2] > 1.85 and a[2] > 0:
            a[2] = np.random.uniform(0.9 * (1.85 - a[1]), 1.85 - a[1])
        if type(a) == np.ndarray:
            a = a.tolist()
        a.extend([0.0]*3)
        return a


if "__name__" == "__main__":
    enV = Env
