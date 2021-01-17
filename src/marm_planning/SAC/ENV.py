#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy
import moveit_commander
import sys
import os


class Env:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.target_position = [0.5] * 3
        self.r = 0.0
        self.success_dist = 0.05
        self.success_dist_min = 0.05
        self.link = 'link_6'
        self.link_pos = [0.0] * 3
        self.random_pos = []
        self.a_bound = [2.9, 1.88, 2.2]
        self.offset = [0.0, 0.20, -1.1]
        # self.a_bound = [0.0, 1.88, 2.2]
        # self.offset = [0.0, 0.20, -1.1]
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)
        # 初始化ROS节点
        rospy.init_node('moveit_fk_demo', anonymous=True)
        # 初始化需要使用move group控制的机械臂中的arm group
        self.arm = moveit_commander.MoveGroupCommander('arm')
        # 设置目标位置所使用的参考坐标系
        self.reference_frame = 'base_link'
        self.arm.set_pose_reference_frame(self.reference_frame)

    def get_random_pos(self):
        a = np.random.uniform(-1, 1, 3) * self.a_bound
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
        print "target has updated to:" + str(pos)
        new_line = self.target_position
        with open('object.txt', 'r+') as f, open('object.urdf', 'w')as q:
            count = 0
            for line in f:
                count += 1
                if count == 9:
                    line = line.replace(line[19:-4], str(new_line[0]) + ' '
                                        + str(new_line[1]) + ' ' + str(new_line[2]-0.1))
                q.write(line)
        os.system('rosservice call gazebo/delete_model "model_name: my_box"')
        os.system('rosrun gazebo_ros spawn_model '
                  '-file /home/you/robotiq/src/marm_planning/SAC/object.urdf -urdf -z 1 -model my_box')

    def action(self, action):
        action = self.action_limit(action)
        self.arm.set_joint_value_target(action)
        self.arm.set_start_state_to_current_state()
        self.arm.go()
        pose = self.arm.get_current_pose(self.link)
        self.link_pos[0] = round(pose.pose.position.x - self.target_position[0], 3)
        self.link_pos[1] = round(pose.pose.position.y - self.target_position[1], 3)
        self.link_pos[2] = round(pose.pose.position.z - self.target_position[2]-0.8, 3)

    def reset(self, reset_action):
        self.steps = 0
        self.done = False
        reset_action = reset_action[:]
        state = []
        state.extend(reset_action)
        self.action(reset_action)
        # state.extend(self.link_pos)
        state.extend(self.target_position)
        return state

    def step(self, action):
        self.steps += 1
        state = []
        state.extend(action)
        self.action(action)
        # state.extend(self.link_pos)
        state.extend(self.target_position)
        dist_new = pow(pow(self.link_pos[0], 2) + pow(self.link_pos[1], 2) + pow(self.link_pos[2], 2), .5)
        # display the distance every 20 steps
        if self.steps % 20 == 0:
            print 'dist:' + str(round(dist_new, 3))
        self.r = (np.exp(-0.99*dist_new)-1)*20
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
