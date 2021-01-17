#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/JointState.h"
/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
 
void jointstatesCallback(const sensor_msgs::JointStateConstPtr& msg)
{
    float pos[8],vel[8],eff[8];
 // pos=msg.position;
	for(int i=0;i<8;i++){
	pos[i] = msg->position[i];
	//vel[i] = msg->velocity[i];
	//eff[i] = msg->effort[i];
	}

  	ROS_INFO("I heard the position of the joints: [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]",pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],pos[7]);
  	//ROS_INFO("the velocity of the joints: [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]",vel[0],vel[1],vel[2],vel[3],vel[4],vel[5],vel[6],vel[7]);
   	//ROS_INFO("and the effort of the joints: [%f] [%f] [%f] [%f] [%f] [%f] [%f] [%f]",eff[0],eff[1],eff[2],eff[3],eff[4],eff[5],eff[6],eff[7]);
}
 
int main(int argc, char **argv)
{
 
  ros::init(argc, argv, "joint_states_listener");
 
 
  ros::NodeHandle n;
 
 
  ros::Subscriber sub = n.subscribe("/joint_states", 1, jointstatesCallback);
 
 
  ROS_INFO("NOW,I'M LISTENING TO THE JOINT_STATES!");


  //ros::Rate loop_rate(1);
  
  //while (ros::ok()){
    //ros::spinOnce();                 
    //loop_rate.sleep();
//}
    ros::spin();
 
 
  return 0;
}
