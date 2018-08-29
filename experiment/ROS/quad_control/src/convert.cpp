#include <ros/ros.h>
#include <bits/stdc++.h>
#include <geometry_msgs/Transform.h>
#include <asctec_hl_comm/mav_ctrl.h>

using namespace ros;

Publisher pub;
Subscriber sub;

void callback(const geometry_msgs::Transform &msg) {
    ROS_WARN("HbirdB:= x: %.2f, y: %.2f, z: %.2f, yaw: %.2f, pitch: %.2f",msg.translation.x, msg.translation.y, msg.translation.z, msg.rotation.x, msg.rotation.y, msg.rotation.z);

    // convert and publish control message
    asctec_hl_comm::mav_ctrl control;
    control.type = 2;
    control.x = msg.translation.x;
    control.y = msg.translation.y;
    control.z = msg.translation.z;
    control.yaw = msg.rotation.z;
    control.v_max_xy = 1;
    control.v_max_z = 1;
		pub.publish(control);
}

int main(int argc, char **argv) {
    init(argc, argv, "convert");
  	NodeHandle nh;

  	sub = nh.subscribe("/controller", 10000, &callback);
  	pub = nh.advertise<asctec_hl_comm::mav_ctrl>("/cov_ctrl_1", 10000);
  	spin();
}
