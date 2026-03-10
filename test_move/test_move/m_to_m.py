#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist


import numpy as np
class MtoM (Node):
    def __init__(self):
        super().__init__("m_to_m")

        self.R_wheel = 0.0535 # unit in meter
        self.wheel_distance = 0.34 # unit in meter
        
        self.msg_rpm = Float64MultiArray()
        self.msg_rpm.data = [0.0, 0.0]

        # self.motion_sub = self.create_subscription(Twist, "/PT/motion", self.motion_sub_callback, 10)
        self.motion_sub = self.create_subscription(Twist, "/cmd_vel", self.motion_sub_callback, 10)
        self.move_pub = self.create_publisher(Float64MultiArray, "/PT/move", 10)
        self.timers_ = self.create_timer (0.1, self.move_pub_callback)

        self.get_logger().info("m_to_m soon")

    def motion_sub_callback (self, msg: Twist):

        l_rpm = (msg.linear.x + (self.wheel_distance / 2) * (msg.angular.z)) / self.R_wheel
        r_rpm = ((msg.linear.x - (self.wheel_distance / 2) * (msg.angular.z)) / self.R_wheel)
        l_rpm = l_rpm *(60/(2*np.pi))
        r_rpm = -r_rpm *(60/(2*np.pi))
        
        self.msg_rpm.data = [l_rpm, r_rpm]
        print ("l_rpm-r_rpm: ", self.msg_rpm.data[0], "-", self.msg_rpm.data[1])
    
    def move_pub_callback (self):
        self.move_pub.publish (self.msg_rpm)

def main (args=None):
    rclpy.init(args=args)
    node = MtoM ()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()