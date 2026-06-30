#!/usr/bin/env python3
import rclpy
import numpy as np
import math
import heapq
from custom_interfaces.msg import Path as PathMsg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from custom_interfaces.srv import HybridAStar
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from scipy.ndimage import distance_transform_edt
from rclpy.qos import QoSProfile, DurabilityPolicy


class node_state:
    def __init__(self, x, y, theta, v = 0.0, steer = 0.0, g=0.0, parent=None, state = 1):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.steer = steer
        self.g = g
        self.parent = parent
        self.state = state

    def key(self):
        theta_bins = 24
        theta_normalized = (self.theta + np.pi) % (2 * np.pi)

        theta_disc = int(theta_normalized / (2 * np.pi) * theta_bins)
        return (
            round((self.x+3.0)/0.04),
            round((self.y+2.5)/0.04),
            theta_disc,
            self.state
        )

 
class HybridAStarNode(Node): 
    def __init__(self):
        super().__init__("hybrid_a_star") 

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        #vehicle parameters
        self.wheelbase_ = 0.22
        self.max_steering_angle_Right = np.deg2rad(10)
        self.max_steering_angle_left = -np.deg2rad(18.75)
        self.max_vel_ =  1.0
        self.min_vel_ =  0.5
        self.dt_ = 0.1
        
        self.open_set_ = {}
        self.closed_set_ = set()


        self.Start_ = [0.0, 0.0, 0.0]
        self.Goal_ = [0.0, 0.0, 0.0]
        self.cached_goal_ = self.Goal_
        self.Grid_ = np.load("/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/occupancy_grid.npy")
        self.distance_map = distance_transform_edt(1 - self.Grid_)
        self.path = PathMsg()

        self.map_resolution_  = 0.01
        self.map_origin_x_ = -3.0
        self.map_origin_y_ = -2.5

        # motion primitives

        self.steer_inputs = np.linspace(self.max_steering_angle_left, self.max_steering_angle_Right, 5).tolist()

        self.velocity_inputs = [
            self.max_vel_,
            -self.min_vel_
            # (self.max_vel_ + self.min_vel_)/2
        ]

        self.sub_goal_ = self.create_service(HybridAStar, "/goal", self.goal_callback)

        self.sub_odom_ = self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.get_logger().info("Hybrid A* node has been started.")

        self.path_pub_ = self.create_publisher(PathMsg, "/path", path_qos)
        self.goal_point_pub_ = self.create_publisher(PoseStamped, "/goal_point", 10)

        self.timer_ = self.create_timer(0.1, self.publish_path)

    def compute_heuristic_map(self, goal_x, goal_y):

        h_map = np.full(self.Grid_.shape,np.inf,dtype=np.float32)

        pq = []

        heapq.heappush(pq,(0.0, goal_x, goal_y))

        h_map[goal_y, goal_x] = 0.0

        motions = [(1,0,1),(-1,0,1),(0,1,1),(0,-1,1)]

        while pq:

            cost, x, y = heapq.heappop(pq)

            if cost > h_map[y, x]:
                continue

            for dx, dy, c in motions:

                nx = x + dx
                ny = y + dy

                if nx < 0 or nx >= self.Grid_.shape[1]:
                    continue

                if ny < 0 or ny >= self.Grid_.shape[0]:
                    continue

                if self.Grid_[ny, nx]:
                    continue

                move_cost = c

                new_cost = cost + move_cost

                if new_cost < h_map[ny, nx]:

                    h_map[ny, nx] = new_cost

                    heapq.heappush(
                        pq,
                        (new_cost, nx, ny)
                    )

        return h_map
    
    def world_to_grid(self, x, y):
        gx = int((x - self.map_origin_x_) / self.map_resolution_)
        gy = int((y - self.map_origin_y_) / self.map_resolution_)
        return gx, gy

    def grid_to_world(self, gx, gy):
        x = gx * self.map_resolution_ + self.map_origin_x_
        y = gy * self.map_resolution_ + self.map_origin_y_
        return x, y

    def heuristic(self, x, y):

        x_idx ,y_idx = self.world_to_grid(x, y)

        if x_idx < 0 or x_idx >= self.Grid_.shape[1]:
            return np.inf

        if y_idx < 0 or y_idx >= self.Grid_.shape[0]:
            return np.inf

        return self.h_map[y_idx, x_idx]
    
    def euler_plus_heading(self, x, y, theta):

        cost = np.hypot((self.Goal_[0]-x),(self.Goal_[1]-y))

        cost = cost + np.arctan2(np.sin(self.Goal_[2]-theta),np.cos(self.Goal_[2]-theta))

        return cost
    
    def trajectory_collision(self, x, y, theta, v, s):

        step_size = 0.02
        steps = min(int(self.dt_ / step_size), 7)
        
        x_curr = x
        y_curr = y
        theta_curr = theta

        for _ in range(steps):

            x_curr, y_curr, theta_curr = self.runge_kutta(x_curr,y_curr,theta_curr,v,s,step_size)

            theta_curr = np.arctan2(np.sin(theta_curr),np.cos(theta_curr))

            if self.collision(x_curr, y_curr, theta_curr):
                return True

        return False
    
    def collision(self, x, y, theta):

        front_offset = 0.09
        rear_offset = -0.09
        side_offset = 0.025

        front_x = x + front_offset * np.cos(theta)
        front_y = y + front_offset * np.sin(theta)

        rear_x = x + rear_offset * np.cos(theta)
        rear_y = y + rear_offset * np.sin(theta)

        Lside_x = x - side_offset * np.sin(theta)  #0.15 also works for path to be more in center
        Lside_y = y + side_offset * np.cos(theta)

        Rside_x = x + side_offset * np.sin(theta)
        Rside_y = y - side_offset * np.cos(theta)

        FL_x = x + front_offset * np.cos(theta) - side_offset * np.sin(theta)
        FL_y = y + front_offset * np.sin(theta) + side_offset * np.cos(theta)
        FR_x = x + front_offset * np.cos(theta) + side_offset * np.sin(theta)
        FR_y = y + front_offset * np.sin(theta) - side_offset * np.cos(theta)
        RL_x = x + rear_offset  * np.cos(theta) - side_offset * np.sin(theta)
        RL_y = y + rear_offset  * np.sin(theta) + side_offset * np.cos(theta)
        RR_x = x + rear_offset  * np.cos(theta) + side_offset * np.sin(theta)
        RR_y = y + rear_offset  * np.sin(theta) - side_offset * np.cos(theta)

        return (self.point_collision(front_x, front_y) or 
                self.point_collision(rear_x, rear_y) or 
                self.point_collision(Lside_x, Lside_y) or 
                self.point_collision(Rside_x, Rside_y) or
                self.point_collision(FL_x, FL_y) or
                self.point_collision(FR_x, FR_y) or
                self.point_collision(RL_x, RL_y) or
                self.point_collision(RR_x, RR_y)
                )

    def point_collision(self, x, y):

        x_idx ,y_idx = self.world_to_grid(x, y)

        if x_idx < 0 or x_idx >= self.Grid_.shape[1]:
                return True

        if y_idx < 0 or y_idx >= self.Grid_.shape[0]:
                return True

        if self.distance_map[y_idx, x_idx] < 0.06/self.map_resolution_ :
            return True

        return False

    def clearance_cost(self, x, y):

        x_idx ,y_idx = self.world_to_grid(x, y)

        dist = self.distance_map[y_idx, x_idx]*self.map_resolution_
        return 2/(dist + 0.01)
        
    def runge_kutta(self, x, y, theta, v, s, h):


            def rk4(x, y, theta):

                dx = v * np.cos(theta)
                dy = v * np.sin(theta)
                dtheta = (v / self.wheelbase_) * np.tan(s)

                return dx, dy, dtheta

            k1_x, k1_y, k1_theta = rk4(x, y, theta)

            k2_x, k2_y, k2_theta = rk4(
                x + 0.5*h*k1_x,
                y + 0.5*h*k1_y,
                theta + 0.5*h*k1_theta
            )

            k3_x, k3_y, k3_theta = rk4(
                x + 0.5*h*k2_x,
                y + 0.5*h*k2_y,
                theta + 0.5*h*k2_theta
            )

            k4_x, k4_y, k4_theta = rk4(
                x + h*k3_x,
                y + h*k3_y,
                theta + h*k3_theta
            )

            x_new = x + (h/6.0)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_new = y + (h/6.0)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
            theta_new = theta + (h/6.0)*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)

            return x_new,y_new,theta_new
            

    def hybrid_a_star(self, start, goal):

        self.open_set_ = []
        self.closed_set_ = set()

        start_node = node_state(start[0], start[1], start[2],0, 0, 0, None, 1)

        heapq.heappush(self.open_set_, (0, id(start_node), start_node))

        best_cost = {}
        best_cost[start_node.key()] = 0

        expansions = 0

        while self.open_set_:
            _, _, current = heapq.heappop(self.open_set_)

            current_key = current.key()

            if current_key in self.closed_set_:
                continue

            self.closed_set_.add(current_key)

            expansions += 1

            # if expansions % 1000 == 0:
            #     print(f"Expansions: {expansions}, Open set size: {len(self.open_set_)}, Closed set size: {len(self.closed_set_)}")

            # Goal check
            if np.hypot(current.x - goal[0], current.y - goal[1]) < 0.05 and abs(np.arctan2(np.sin(current.theta - goal[2]), np.cos(current.theta - goal[2]))) < np.deg2rad(25):

                print("Goal reached!")
                print(f"Expansions: {expansions}, Open set size: {len(self.open_set_)}, Closed set size: {len(self.closed_set_)}")

                path = []

                node = current

                while node is not None:
                    path.append((node.x, node.y, node.theta))
                    node = node.parent

                return path[::-1]

            for s in self.steer_inputs:
                for v in self.velocity_inputs:

                    x_new, y_new, theta_new = self.runge_kutta(current.x, current.y, current.theta, v, s, self.dt_)
                    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

                    collision_found = self.trajectory_collision(current.x, current.y, current.theta, v, s)
                    if collision_found == True:
                        continue

                    if not (current.v == v ):
                        reversing_cost = 20
                    else:
                        reversing_cost = 0

                    if v < 0:
                        state = 0
                        w = 2.0
                    else:
                        state = 1
                        w = 1.0

                    c = self.clearance_cost(x_new, y_new)
                    h_cost = max(self.heuristic(x_new,y_new), self.euler_plus_heading(x_new,y_new,theta_new))
                    g_cost = current.g + self.dt_ * abs(v) * w + abs(s)*0.1 + c + abs(s - current.steer)*0.2 + reversing_cost

                    neighbor = node_state(x_new, y_new, theta_new, v,s, g_cost, current,state)
                    neighbor_key = neighbor.key()

                    if neighbor_key not in best_cost or g_cost < best_cost[neighbor_key]:
                        best_cost[neighbor_key] = g_cost
                        f_cost = g_cost + h_cost
                        heapq.heappush(self.open_set_, (f_cost, id(neighbor), neighbor))

        print("no path found")
        return False


        
    def goal_callback(self, req: HybridAStar.Request, res: HybridAStar.Response):
        
        self.Goal_ = req.goal

        pose = PoseStamped()
        pose.header.frame_id = "odom"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.Goal_[0]
        pose.pose.position.y = self.Goal_[1]
        pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, self.Goal_[2])
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        self.goal_point_pub_.publish(pose)

        if self.Grid_ is None:
            self.get_logger().warn("Occupancy grid map not received yet. Cannot plan path.")
            res.feedback = "Occupancy grid map not received yet. Cannot plan path."
            return res
        if self.Start_ is None:
            self.get_logger().warn("Odometry data not received yet. Cannot plan path.")
            res.feedback = "Odometry data not received yet. Cannot plan path."
            return res
        change_in_goal_dist = np.hypot((self.Goal_[0]-self.cached_goal_[0]),(self.Goal_[1]-self.cached_goal_[1]))
        
        g_x, g_y = self.world_to_grid(self.Goal_[0], self.Goal_[1])
        s_x, s_y = self.world_to_grid(self.Start_[0], self.Start_[1])

        if change_in_goal_dist >= 0.1:
            self.h_map = self.compute_heuristic_map(g_x, g_y)
            self.cached_goal_ = self.Goal_

        if self.collision(self.Goal_[0], self.Goal_[1], self.Goal_[2]) or self.Grid_[s_y, s_x]:
            self.get_logger().warn("Start or goal position is in collision. Cannot plan path.")
            res.feedback = "Start or goal position is in collision. Cannot plan path."
            return res

        self.get_logger().info(f"Planning started .....")
        temp = self.hybrid_a_star(self.Start_, self.Goal_)

        if not temp:
            self.get_logger().warn("Path planning failed. No path found.")
            res.feedback = "Path planning failed. No path found."
            return res
        
        self.path.x = [p[0] for p in temp]
        self.path.y = [p[1] for p in temp]
        self.path.theta = [p[2] for p in temp]
        self.get_logger().info("Path planning successful. Path published.")
        res.feedback = "Path planning successful. Path published."
        return res


    def odom_callback(self, msg: Odometry):
    
        [_,_,yaw] = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.Start_ = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def publish_path(self):
        if len(self.path.x) > 0:
            self.path_pub_.publish(self.path)
            # temp = self.hybrid_a_star(self.Start_, self.Goal_)
            # if not temp:
            #     pass
            # else:
            #     self.path.x = [p[0] for p in temp]
            #     self.path.y = [p[1] for p in temp]
            #     self.path.theta = [p[2] for p in temp]
 
 
def main(args=None):
    rclpy.init(args=args)
    node = HybridAStarNode() 
    rclpy.spin(node)
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()