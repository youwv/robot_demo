import socket
import rospy, sys
import moveit_commander

# path = "/home/you/data3.csv"
# head = ["posx", "posy", "posz", "orientationx", "orientationy", "orientationz", "orientationw"]
# b.create_csv(path, head)
# b.append_csv(path, [pos_orientation])
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
    arm.set_goal_joint_tolerance(0.05)

    # 控制机械臂先回到初始化位置
    arm.set_named_target('home')
    arm.go()
    rospy.sleep(2)
    pos_orientation = [0, 0, 0, 0, 0, 0, 0]
    while True:
        client, addr = ser.accept()
        print('accept %s connect' % (addr,))
        data = client.recv(1024)
        data = str(data, encoding='utf-8')
        print(data)
        # 控制机械臂完成运动
        arm.set_joint_value_target(data)
        arm.go()
        rospy.sleep(1)
        pose = arm.get_current_pose(end_effector_link)
        pos_orientation[0] = pose.pose.position.x
        pos_orientation[1] = pose.pose.position.y
        pos_orientation[2] = pose.pose.position.z
        pos_orientation[3] = pose.pose.orientation.x
        pos_orientation[4] = pose.pose.orientation.y
        pos_orientation[5] = pose.pose.orientation.z
        pos_orientation[6] = pose.pose.orientation.w
        client.send(bytes(pos_orientation, encoding='utf-8'))
        client.close()


server()
