#!/usr/bin/env python3
from sympy import euler

import rclpy
import math
from rclpy.node import Node
from rclpy.time import Time
from custom_interfaces.msg import Path as PathMsg
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

class TFPublisheriitb(Node): 
    def __init__(self):
        super().__init__("tf_publisher")

        self.declare_parameter("axel_length",0.26)
        self.L = self.get_parameter("axel_length").value    

        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "base_footprint"
        self.odom_msg.header.stamp = self.get_clock().now().to_msg()

        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0

        self.odom_msg.pose.pose.orientation.x = 0.0
        self.odom_msg.pose.pose.orientation.y = 0.0
        self.odom_msg.pose.pose.orientation.z = 0.0
        self.odom_msg.pose.pose.orientation.w = 1.0

        self.xpos = -1.6
        self.ypos = -1.4
        self.theta = 1.57
        self.quat = quaternion_from_euler(0, 0, self.theta)
        self.vel = 0.0
        self.steer = 0.0

        self.tf_broadcaster = TransformBroadcaster(self)
        self.transform_stamped = TransformStamped()
        self.transform_stamped.header.frame_id = "odom"
        self.transform_stamped.child_frame_id = "base_footprint"


        self.create_timer(0.02, self.publish_tf)

        self.subscription_ = self.create_subscription(Float64MultiArray,"/velocity_steer",self.callback,10)
        self.odom_publisher_ = self.create_publisher(Odometry,"odom/sim",10)
        self.sub_ = self.create_subscription(PathMsg,"/path",self.visualize,10)
        self.path_pub = self.create_publisher(Path, "/path_vis", 10)
        self.marker_pub = self.create_publisher(Marker, "/car_sim_bounds_marker", 10)

    def callback(self,msg: Float64MultiArray):
        self.vel = msg.data[0]
        self.steer = msg.data[1]

        if self.steer >= 0:
            self.steer = min(self.steer, math.radians(10.0))
        else:
            self.steer = max(self.steer, math.radians(-18.5))

    def publish_car_box_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"  # Matches your odom_msg frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "car_footprint"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Center the box on the current x/y position
        marker.pose.position.x = self.xpos
        marker.pose.position.y = self.ypos
        marker.pose.position.z = 0.05  # Floating slightly above ground

        # Apply the current fused quaternion for accurate rotation
        marker.pose.orientation.x = self.quat[0]
        marker.pose.orientation.y = self.quat[1]
        marker.pose.orientation.z = self.quat[2]
        marker.pose.orientation.w = self.quat[3]

        # 21 cm (x), 17 cm (y), and an arbitrary 10 cm height (z)
        marker.scale.x = 0.21
        marker.scale.y = 0.17
        marker.scale.z = 0.10

        # Set color to a semi-transparent green
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9  # Alpha (transparency)

        self.marker_pub.publish(marker)


    def publish_tf(self):
        
        current_time = self.get_clock().now()
        last_time = Time.from_msg(self.odom_msg.header.stamp)
        dt = (current_time - last_time).nanoseconds / 1e9

        if dt == 0:
            return
        if dt >= 0.5:
            return

        self.odom_msg.header.stamp = current_time.to_msg()

        # ackerman model
        self.xpos += self.vel * math.cos(self.theta) * dt
        self.ypos += self.vel * math.sin(self.theta) * dt
        self.theta += (self.vel / self.L) * math.tan(self.steer) * dt
        self.quat = quaternion_from_euler(0, 0, self.theta)


        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_msg.pose.pose.position.x = self.xpos
        self.odom_msg.pose.pose.position.y = self.ypos
        self.odom_msg.pose.pose.position.z = 0.0

        self.odom_msg.pose.pose.orientation.x = self.quat[0]
        self.odom_msg.pose.pose.orientation.y = self.quat[1]
        self.odom_msg.pose.pose.orientation.z = self.quat[2]
        self.odom_msg.pose.pose.orientation.w = self.quat[3]
        self.odom_msg.twist.twist.linear.x = self.vel
        self.odom_msg.twist.twist.angular.z = (self.vel / self.L) * math.tan(self.steer)
        self.odom_publisher_.publish(self.odom_msg)

        self.transform_stamped.header.stamp = self.get_clock().now().to_msg()
        self.transform_stamped.transform.translation.x = self.xpos
        self.transform_stamped.transform.translation.y = self.ypos
        self.transform_stamped.transform.translation.z = 0.0
        self.transform_stamped.transform.rotation.x = self.quat[0]
        self.transform_stamped.transform.rotation.y = self.quat[1]
        self.transform_stamped.transform.rotation.z = self.quat[2]
        self.transform_stamped.transform.rotation.w = self.quat[3]

        self.tf_broadcaster.sendTransform(self.transform_stamped)


        self.publish_car_box_marker()
        


    def visualize(self, msg: PathMsg):
        
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        path_x = msg.x
        path_y = msg.y
        path_theta = msg.theta

        for i in range(len(path_x)):

            pose = PoseStamped()

            pose.header.frame_id = "odom"
            pose.header.stamp = self.get_clock().now().to_msg()

            pose.pose.position.x = path_x[i]
            pose.pose.position.y = path_y[i]
            pose.pose.position.z = 0.0

            pose.pose.orientation.w = path_theta[i]

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)




def main(args=None):
    rclpy.init(args=args)
    node = TFPublisheriitb() 
    rclpy.spin(node)
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
