#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import LinkStates
import os
from combination import *
bridge = CvBridge()


class Env:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.target_position = [0.5] * 3
        self.r = 0.0
        self.success_dist = 0.1
        self.success_dist_min = 0.1
        self.end = 'panda_hand'
        self.end_pos = [0.0] * 3
        self.random_pos = []
        self.a_bound = [2.9, 1.88, 2.2]
        self.offset = [0.0, 0.20, -1.1]
        self.count = 0
        self.offline = False
        self.node = rospy.init_node('action_reaction', anonymous=True)
        self.person_info_pub1 = rospy.Publisher('/abb/joint_position_controller/command',
                                                Float64MultiArray, queue_size=2)
        self.data1d_action = Float64MultiArray()
        os.system('rosrun gazebo_ros spawn_model '
                  '-file /home/you/robotiq/src/marm_planning/test_version/object.urdf -urdf -model my_box')

    def get_random_pos(self):
        a = np.random.uniform(-1, 1, 3) * self.a_bound / 2
        a += self.offset
        self.data1d_action.data = self.action_limit(a)
        self.person_info_pub1.publish(self.data1d_action)
        link_state = rospy.wait_for_message("/gazebo/link_states", LinkStates)
        self.end_pos[0] = link_state.__getstate__()[1][-2].position.x
        self.end_pos[1] = link_state.__getstate__()[1][-2].position.y
        self.end_pos[2] = link_state.__getstate__()[1][-2].position.z
        self.random_pos.append(self.end_pos[:])

    # def set_target_position(self):
    #     np.random.shuffle(self.random_pos)
    #     pos = self.random_pos[0][:]
    #     self.target_position = pos
    #     os.system("rosservice call /gazebo/set_model_state '{model_state: { model_name: my_box, pose: "
    #               "{ position: { x: %f, y: %f ,z: %f }} , reference_frame: world } }'"
    #               % (float(pos[0]), float(pos[1]), float(pos[2])))
    #     print "target has updated to:" + str(pos)

    def set_target_position(self):
        np.random.shuffle(self.random_pos)
        pos = self.random_pos[0][:]
        self.target_position = pos
        new_line = self.target_position
        with open('object.txt', 'r+') as f, open('object.urdf', 'w')as q:
            count = 0
            for line in f:
                count += 1
                if count == 9:
                    line = line.replace(line[19:-4], str(new_line[0]) + ' '
                                        + str(new_line[1]) + ' ' + str(new_line[2]))
                q.write(line)
        os.system('rosservice call gazebo/delete_model "model_name: my_box"')
        os.system('rosrun gazebo_ros spawn_model '
                  '-file /home/you/robotiq/src/marm_planning/test_version/object.urdf -urdf -model my_box')
        print "target has updated to:" + str(pos)

    def action(self, action):
        self.data1d_action.data = self.action_limit(action)
        self.person_info_pub1.publish(self.data1d_action)
        link_state = rospy.wait_for_message("/gazebo/link_states", LinkStates)
        self.end_pos[0] = link_state.__getstate__()[1][-2].position.x
        self.end_pos[1] = link_state.__getstate__()[1][-2].position.y
        self.end_pos[2] = link_state.__getstate__()[1][-2].position.z

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
        dist_new = pow(pow(self.end_pos[0]-self.target_position[0], 2) + pow(self.end_pos[1]-self.target_position[1], 2)
                       + pow(self.end_pos[2]-self.target_position[2], 2), .5)
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
# import rospy
# from std_msgs.msg import Float64MultiArray
# from gazebo_msgs.msg import LinkStates
#
# rospy.init_node('action_reaction', anonymous=True)
#
# person_info_pub1 = rospy.Publisher('/abb/joint_position_controller/command', Float64MultiArray, queue_size=1)
#
# data1 = [-0.0, -0.5, -0.5, -0.5, -0.5, -0.5]
# data1d = Float64MultiArray()
# data1d.data = data1
# person_info_pub1.publish(data1d)
# rospy.loginfo("Publish person message:% s", data1)
# y = rospy.wait_for_message("/gazebo/link_states", LinkStates)
# print y.__getstate__()
