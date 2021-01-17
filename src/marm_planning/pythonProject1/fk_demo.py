#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import rospy
import sys
import moveit_commander


C = 3
# 关闭并退出moveit
# moveit_commander.roscpp_shutdown()
# moveit_commander.os._exit(0)


def server():
    ser = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ser.bind(('127.0.0.1', 7777))
    ser.listen(5)
    # 初始化move_group的API
    moveit_commander.roscpp_initialize(sys.argv)

    # 初始化ROS节点
    rospy.init_node('moveit_fk_demo', anonymous=True)

    # 初始化需要使用move group控制的机械臂中的arm group
    arm = moveit_commander.MoveGroupCommander('arm')

    # 获取终端link的名称
    end_effector_link = arm.get_end_effector_link()

    # 设置目标位置所使用的参考坐标系
    reference_frame = 'base_link'
    arm.set_pose_reference_frame(reference_frame)

    # 设置机械臂和夹爪的允许误差值
    # arm.set_goal_joint_tolerance(0.01)

    # 控制机械臂先回到初始化位置
    # arm.set_named_target('home')
    # arm.go()
    # rospy.sleep(2)
    pos_orientation = [0, 0, 0, 0, 0, 0, 0]
    while True:
        client, addr = ser.accept()
        print 'accept %s connect' % (addr,)
        data = client.recv(1024)
        data = str(data)
        data = data[1:-1]
        data = data.split(",")
        data = map(float, data)
        print data
        # 控制机械臂完成运动
        arm.set_joint_value_target(data)
        arm.go()
        rospy.sleep(2)
        pose = arm.get_current_pose(end_effector_link)
        pos_orientation[0] = round(pose.pose.position.x, 3)
        pos_orientation[1] = round(pose.pose.position.y, 3)
        pos_orientation[2] = round(pose.pose.position.z, 3)
        pos_orientation[3] = round(pose.pose.orientation.x, 3)
        pos_orientation[4] = round(pose.pose.orientation.y, 3)
        pos_orientation[5] = round(pose.pose.orientation.z, 3)
        pos_orientation[6] = round(pose.pose.orientation.w, 3)
        client.send(bytes(pos_orientation))
        client.close()


server()

