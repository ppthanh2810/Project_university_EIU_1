#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.time import Time
from rclpy.duration import Duration
import math 

class Odom (Node):
    def __init__ (self):
        super().__init__("odom")
        self.initialized = False
        
        self.odom_sub_ = self.create_subscription(JointState, "/PT/odom", self.odom_sub_callback, 10)
        
        self.get_logger().info("odom soon")
        
        self.wheel_radius = 0.0535
        self.wheel_separator = 0.34
        
        self.prev_time_ = self.get_clock().now()
        self.left_wheel_prev_pos_ = 0.0
        self.right_wheel_prev_pos_ = 0.0
        self.x_ = 0.0
        self.y_ = 0.0
        self.theta_ = 0.0
        self.linear_filtered = 0.0
        self.angular_filtered = 0.0
        
    def odom_sub_callback(self, msg: JointState):
        # print (f"{msg.position[0]: .2f}-{msg.position[1]: .2f}")
        if not self.initialized:
            self.left_wheel_prev_pos_ = msg.position[0]
            self.right_wheel_prev_pos_ = msg.position[1]
            self.prev_time_ = Time.from_msg(msg.header.stamp)
            self.initialized = True
            self.get_logger().info("Đã đồng bộ vị trí encoder ban đầu!")
            return
       
        if hasattr(self, "block_update_until") and self.get_clock().now() < self.block_update_until:
            return

        if not msg.position or len(msg.position) < 2:
            return

        if any(math.isnan(pos) or math.isinf(pos) for pos in msg.position):
            return

        # Tính dt (delta thời gian giữa hai lần cập nhật)
        current_time = Time.from_msg(msg.header.stamp)
        dt = max((current_time - self.prev_time_).nanoseconds / 1e9, 1e-6)  # Đảm bảo dt >= 1µs
        self.prev_time_ = current_time

        if msg.position[0] == 0.0 and msg.position[1] == 0.0:
            if self.left_wheel_prev_pos_ != 0.0 or self.right_wheel_prev_pos_ != 0.0:
                self.reset_odom()
            return 

        dp_left = msg.position[0] - self.left_wheel_prev_pos_
        dp_right = -(msg.position[1] - self.right_wheel_prev_pos_)
        # print (f"{dp_left: .4f}, {dp_right: .4f}")

        # Ngưỡng bỏ qua nhiễu nhỏ
        if abs(dp_left) < 0.0005 and abs(dp_right) < 0.0005:
            return 

        self.left_wheel_prev_pos_ = msg.position[0]
        self.right_wheel_prev_pos_ = msg.position[1]

        phi_left = dp_left / dt
        phi_right = dp_right / dt

        self.update_odometry(phi_left, phi_right, dt)
        
    def update_odometry(self, phi_left, phi_right, dt):

        self.linear_velocity = (phi_left + phi_right) / 2.0
        self.angular_velocity = (phi_right - phi_left) / self.wheel_separator
        # print (f"{self.linear_velocity: .2f}-{self.angular_velocity: .2f}")

        self.alpha = 0.2
        self.linear_filtered = self.alpha * self.linear_velocity + (1 - self.alpha) * self.linear_filtered
        self.angular_filtered = self.alpha * self.angular_velocity + (1 - self.alpha) * self.angular_filtered

        if (abs(self.linear_filtered) < 0.004):
            self.linear_filtered = 0.0
        if (abs(self.angular_filtered) < 0.004):
            self.angular_filtered = 0.0
        
        # print (f"{self.linear_filtered: .2f}-{self.angular_filtered: .2f}")

        self.x_ += self.linear_filtered * math.cos(self.theta_) * dt
        self.y_ += self.linear_filtered * math.sin(self.theta_) * dt

        self.theta_ += self.angular_filtered * dt
        self.theta_ = math.atan2(math.sin(self.theta_), math.cos(self.theta_))
        
        print (f"{self.x_: .2f}-{self.y_: .2f}-{self.theta_: .2f}")
    
    def reset_odom(self):
        self.x_ = 0.0
        self.y_ = 0.0
        self.theta_ = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.left_wheel_prev_pos_ = 0.0
        self.right_wheel_prev_pos_ = 0.0

        self.block_update_until = self.get_clock().now() + Duration(seconds=0.1)
        self.get_logger().info("Odometry reset")
         
def main (args=None):
    rclpy.init(args=args)
    node = Odom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()