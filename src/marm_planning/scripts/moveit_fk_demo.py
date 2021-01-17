#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import rospy, sys
import moveit_commander
from control_msgs.msg import GripperCommand
import numpy as np


class MoveItFkDemo:
    def __init__(self):
        # 初始化move_group的API
        moveit_commander.roscpp_initialize(sys.argv)

        # 初始化ROS节点
        rospy.init_node('moveit_fk_demo', anonymous=True)

        # 初始化需要使用move group控制的机械臂中的arm group
        arm = moveit_commander.MoveGroupCommander('arm')
        # 初始化需要使用move group控制的机械臂中的gripper group
        # gripper = moveit_commander.MoveGroupCommander('gripper')

        # 获取终端link的名称
        end_effector_link = arm.get_end_effector_link()
                        
        # 设置目标位置所使用的参考坐标系
        reference_frame = 'base_link'
        arm.set_pose_reference_frame(reference_frame)

        # 设置机械臂和夹爪的允许误差值
        arm.set_goal_joint_tolerance(0.05)
        # gripper.set_goal_joint_tolerance(0.001)

        # 控制机械臂先回到初始化位置
        arm.set_named_target('home')
        arm.go()
        rospy.sleep(2)

        # 设置夹爪的目标位置，并控制夹爪运动
        # gripper.set_joint_value_target([0.01])
        # gripper.go()
        # rospy.sleep(1)

        arm.set_joint_value_target([-2.77694057, 2.15, 0.0, 0.0, 0.0, 0.0])
        # 控制机械臂完成运动
        arm.go()
        rospy.sleep(1)
        # 关闭并退出moveit
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)


if __name__ == "__main__":
    try:
        MoveItFkDemo()
    except rospy.ROSInterruptException:
        pass

