#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bridge = CvBridge()
step = 0


def get_picture():
    global step
    step += 1
    data1 = rospy.wait_for_message("/arm/camera1/image_raw", Image)
    data2 = rospy.wait_for_message("/arm/camera2/image_raw", Image)
    left_cv_image = bridge.imgmsg_to_cv2(data1)
    right_cv_image = bridge.imgmsg_to_cv2(data2)
    # x = np.concatenate([left_cv_image, right_cv_image], 1)
    left_cv_image = cv2.cvtColor(left_cv_image, cv2.COLOR_BGR2GRAY)
    right_cv_image = cv2.cvtColor(right_cv_image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(str(step)+'.PNG', x)
    return [left_cv_image, right_cv_image]


if __name__ == '__main__':
    get_picture()
