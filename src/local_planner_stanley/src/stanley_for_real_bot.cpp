#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "custom_interfaces/msg/path.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "nav_msgs/msg/path.hpp"          // <-- ADDED for RViz path visualization
#include "geometry_msgs/msg/pose_stamped.hpp" // <-- ADDED for path poses
#include <cmath>
#include <vector>
#include <iostream>

using namespace std::placeholders;

class StanleyControllerNode : public rclcpp::Node
{
public:
   StanleyControllerNode() : Node("stanley_controller")
   {
       
       rclcpp::QoS qos(1);
       qos.transient_local();
       RCLCPP_INFO(this->get_logger(), "Stanley Controller Node has been started.");
       this->declare_parameter<double>("Proportional", 0.37);
       this->declare_parameter<double>("Integral", 0.00);
       this->declare_parameter<double>("Derivative", 0.10);
       this->declare_parameter<double>("Wheelbase", 0.21);
       
       K_P_ = this->get_parameter("Proportional").as_double();
       K_I_ = this->get_parameter("Integral").as_double();
       K_D_ = this->get_parameter("Derivative").as_double();
       L_ = this->get_parameter("Wheelbase").as_double();

       odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/odometry/filtered", 10, std::bind(&StanleyControllerNode::odomCallback, this,_1));
       control_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/velocity_steer", 10);
       path_sub_ = this->create_subscription<custom_interfaces::msg::Path>("/path", qos, std::bind(&StanleyControllerNode::pathCallback, this,_1));
       twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/test_twist",10);
       
       // <-- ADDED: Publisher for RViz visualization
       robot_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/real_robot_actual_path", 10);
   }
private:
   double K_P_;
   double K_I_;
   double K_D_ ;
   double base_k = 1.0;
   double last_error_ = 0.0;
   double L_;
   double velocity_ ;
   float cross_track_error_;
   float heading_error_;
   double last_time_ = 0.0;

   std::vector<double> path_points_x_;
   std::vector<double> path_points_y_;
   std::vector<double> path_points_theta_;
   std::vector<double> distance;
   double x;
   double y;
   double theta;

   // <-- ADDED: Persistent path message container to store historic steps
   nav_msgs::msg::Path robot_path_history_;

   rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_ ;
   rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr control_pub_ ;
   rclcpp::Subscription<custom_interfaces::msg::Path>::SharedPtr path_sub_ ;
   rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
   
   // <-- ADDED: Publisher object
   rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr robot_path_pub_;

   void pathCallback(const custom_interfaces::msg::Path::SharedPtr msg)
   {
       path_points_x_ = msg->x;
       path_points_y_ = msg->y;
       path_points_theta_ = msg->theta;
   }

   void set_velocity(int index)
    {
        const double base_velocity = 0.7;
        const double min_velocity  = 0.43;
        const int lookahead_step = 3;

        if(index + 3 * lookahead_step >= path_points_x_.size())
        {
            velocity_ = min_velocity;
            return;
        }

        int i1 = index;
        int i2 = index + lookahead_step;
        int i3 = index + 2 * lookahead_step;
        int i4 = index + 3 * lookahead_step;

        double d_theta1 = atan2(path_points_y_[i2] - path_points_y_[i1],path_points_x_[i2] - path_points_x_[i1]);
        double d_theta2 = atan2(path_points_y_[i3] - path_points_y_[i2],path_points_x_[i3] - path_points_x_[i2]);
        double d_theta3 = atan2(path_points_y_[i4] - path_points_y_[i3],path_points_x_[i4] - path_points_x_[i3]);

        double d1 = atan2(sin(d_theta2 - d_theta1),cos(d_theta2 - d_theta1));

        double d2 = atan2(sin(d_theta3 - d_theta2),cos(d_theta3 - d_theta2));

        double total_turning = std::abs(d1) + std::abs(d2);

        double normalized_turn = std::min(total_turning / (M_PI / 2.0),1.0);

        double reduction = normalized_turn * normalized_turn;

        velocity_ = base_velocity - reduction * (base_velocity - min_velocity);

        if(total_turning > 1.2)
        {
            velocity_ *= 0.7;
        }

        velocity_ = std::max(velocity_,min_velocity);
    }

   double getYawFromQuaternion(const geometry_msgs::msg::Quaternion & quat)
   {
       double siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y);
       double cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z);
       return atan2(siny_cosp, cosy_cosp);
   }

   std::pair<double, double> computeError(double x, double y, double theta)
   {
       if (path_points_x_.empty())
       {
           return std::make_pair(0.0, 0.0);
       }
       distance.clear();
       for (size_t i = 0; i < path_points_x_.size(); ++i) {
           distance.push_back(hypot(path_points_x_[i] - x, path_points_y_[i] - y));
       }
      
       int closest_index = std::distance(distance.begin(), std::min_element(distance.begin(), distance.end()));
       int target_index = std::min(closest_index + 3,(int)path_points_x_.size() - 1);

       double closest_x = path_points_x_[target_index];
       double closest_y = path_points_y_[target_index];

      
       set_velocity(closest_index);

       double fx = x + 0.3 * L_ * cos(theta);
       double fy = y + 0.3 * L_ * sin(theta);

       double dx = closest_x - fx;
       double dy = closest_y - fy;

       double perp_x = -sin(theta);
       double perp_y =  cos(theta);

       double error = (dx * perp_x + dy * perp_y);

       double path_theta = path_points_theta_[target_index];
       double heading_error = path_theta - theta;

       heading_error = atan2(sin(heading_error), cos(heading_error));

       return std::make_pair(error, heading_error);
   }

   void computeControlCommand(double cross_track_error, double heading_error)
   {
       double current_time_ = this->now().seconds();
       double dt = current_time_ - last_time_;

       if(dt <= 0.0001 || dt > 0.2)
       {dt = 1.0 / 30.0;}

       last_time_ = current_time_;

       double integral = last_error_ + cross_track_error;
       if (integral > 1.0) {
           integral = 1.0;
       } else if (integral < -1.0) {
           integral = -1.0;
       }
       double derivative = (cross_track_error - last_error_)/dt;
       double control_command = heading_error + atan2((K_P_ * cross_track_error + K_I_ * integral + K_D_ * derivative),velocity_*1.0 + 1e-5);

       std_msgs::msg::Float64MultiArray control_msg;
       control_msg.data = {velocity_, control_command};
    //    control_pub_->publish(control_msg);

        geometry_msgs::msg::Twist twist_msg;

        twist_msg.linear.x = velocity_;
        twist_msg.angular.z = control_command;

        twist_pub_->publish(twist_msg);
   }
  
   void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
   {
       if (path_points_x_.empty()) {
           return;
       }

       x = msg->pose.pose.position.x;
       y = msg->pose.pose.position.y;
       
       // <-- ADDED: Record and Publish the path trajectory data 
       geometry_msgs::msg::PoseStamped current_pose;
       current_pose.header = msg->header; // Inherit timestamp and frame_id (usually 'odom')
       current_pose.pose = msg->pose.pose;
       
       robot_path_history_.header = msg->header;
       robot_path_history_.poses.push_back(current_pose);
       robot_path_pub_->publish(robot_path_history_);
       double dist_to_goal = hypot(x - path_points_x_.back(),y - path_points_y_.back());

       if(dist_to_goal < 0.3) {
           velocity_ = 0.0;
           geometry_msgs::msg::Twist twist_msg;

           twist_msg.linear.x = velocity_;
           twist_msg.angular.z = 0.0;
           twist_pub_->publish(twist_msg);
           return;
       }
       theta = getYawFromQuaternion(msg->pose.pose.orientation);
       
       auto errors = computeError(x, y, theta);
       cross_track_error_ = errors.first;
       heading_error_ = errors.second;

       computeControlCommand(cross_track_error_, heading_error_);
       last_error_ = cross_track_error_;
   }
};
  
int main(int argc, char **argv)
{
   rclcpp::init(argc, argv);
   auto node = std::make_shared<StanleyControllerNode>();
   rclcpp::spin(node);
   rclcpp::shutdown();
   return 0;
}

