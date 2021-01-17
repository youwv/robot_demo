#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import rospy
# import cv2
# import message_filters
# import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import Image
#
# bridge = CvBridge()
# global Picture_Picture
# step = 0
#
#
# def callback(left_image, right_image):
#     global step
#     step += 1
#     left_cv_image = bridge.imgmsg_to_cv2(left_image)
#     right_cv_image = bridge.imgmsg_to_cv2(right_image)
#     # cv2.imshow('left_image', left_cv_image)
#     # cv2.imshow('right_image', right_cv_image)
#     global Picture_Picture
#     x = np.concatenate([left_cv_image, right_cv_image], 1)
#     Picture_Picture = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('111', x)
#     # cv2.waitKey()
#     # print np.shape(x)
#     rospy.signal_shutdown("closed")
#
#
# def get_picture():
#     # rospy.init_node('gazebo_image_sub', anonymous=True)
#     left_ros_image = message_filters.Subscriber("/arm/camera1/image_raw", Image)
#     right_ros_image = message_filters.Subscriber("/arm/camera2/image_raw", Image)
#     ts = message_filters.TimeSynchronizer([left_ros_image, right_ros_image], 10)
#     ts.registerCallback(callback)
#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()
#     return Picture_Picture
# import sys
# print sys.getsizeof(s)
# import numpy as np
# indice = []
# for i in range(15000):
#     indice.append(i)
# indices = np.random.choice(indice[-10000:], size=32)
# print indices
import datadeal
import re
import save_transition
import time

transition_saver = save_transition.Transition_Save(buffer_size=10000)
state, action, reward, next_state, done = transition_saver.dump(batch_size=64)

start_time = time.clock()
for index in range(len(action)):
    data = action[index][1:-1]
    data = re.findall('[^\s]+', data, re.S)
    action[index] = map(float, data)
end_time = time.clock()
spend_time = end_time - start_time
print spend_time
