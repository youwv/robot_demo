#include<iostream>
#include <string>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <robot_state_publisher/robot_state_publisher.h>
using namespace std;

int main(int argc, char** argv)
 {
    ros::init(argc, argv, "joint_states_talker"); //节点的名称
    ros::NodeHandle n;
    ros::Publisher joint_pub = n.advertise<sensor_msgs::JointState>("joint_states", 1);    //设置一个发布者，将消息发布出去,发布到相应的节点中去
    ros::Rate loop_rate(0.1);    //这个设置的太大，效果很不好，目前觉得为10最好了
    const double degree = M_PI/180;
    const double radius = 2;
    int i=-69;

    // robot state
    double angle= 0;
    // message declarations
    sensor_msgs::JointState joint_state;
  //  for(int j=0;j<6;++j)
   // {
        //joint_state.velocity.push_back(1);
        //joint_state.effort.push_back(200);
    //}

   

    while (ros::ok()) {
        //update joint_state
        joint_state.header.stamp = ros::Time::now();
        joint_state.name.resize(8);
        joint_state.position.resize(8);
	    joint_state.name[0]="joint_1";
        joint_state.position[0] = i*degree;
	    joint_state.name[1] ="joint_2";
        joint_state.position[1] = i*degree;
	    joint_state.name[2] ="joint_3";
        joint_state.position[2] = i*degree;
        joint_state.name[3] ="joint_4";
        joint_state.position[3] = i*degree;
	    joint_state.name[4] ="joint_5";
        joint_state.position[4] = i*degree;
	    joint_state.name[5] ="joint_6";
        joint_state.position[5] = i*degree;
	    joint_state.name[6] ="panda_finger_joint1";
        joint_state.position[6] = 0.0;
	    joint_state.name[7] ="panda_finger_joint2";
        joint_state.position[7] = 0.0;

        if(i<=70)
            i++;
        else
            i=-69; 

        //send the joint state and transform
        joint_pub.publish(joint_state);

        // This will adjust as needed per iteration
        loop_rate.sleep();
    }

    return 0;
}

