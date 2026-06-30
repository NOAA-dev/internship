#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from tf_transformations import quaternion_from_euler
from collections import deque
from vesc_msgs.msg import VescStateStamped
from visualization_msgs.msg import Marker

# ==============================================================================
#  STEERING CALIBRATION TABLE  —  Ackermann average of BOTH wheels
#
#  Servo pos  | Right wheel | Left wheel | Average used | Notes
#  -----------+-------------+------------+--------------+-------------------
#  0.20       |  -10°       |  -10°      |  -10.0°      | LEFT hard limit
#  0.35       |   -6°       |   -6°      |   -6.0°      | symmetric
#  0.53       |    0°       |    0°      |    0.0°      | TRUE center
#  0.65       |   +5°       |   +5°      |   +5.0°      | symmetric
#  0.85       |  +18.5°     |  +11°      |  +14.75°     | Ackermann kicks in
#  0.90       |  +22°       |  +15.5°    |  +18.75°     | RIGHT hard limit
#
#  IMPORTANT: The linkage is ASYMMETRIC.
#    Left  side maxes at -10°  (servo 0.53 → 0.20)
#    Right side maxes at +18.75° (servo 0.53 → 0.90)
#
#  Sign convention: positive = right turn, negative = left turn
# ==============================================================================
SERVO_CAL_POS = [0.20,   0.35,  0.53,  0.65,   0.85,    0.90]
RIGHT_CAL_DEG = [-10.0,  -6.0,   0.0,  +5.0,  +18.5,   +22.0]
LEFT_CAL_DEG  = [-10.0,  -6.0,   0.0,  +5.0,  +11.0,   +15.5]
WHEEL_CAL_DEG = [(r + l) / 2.0 for r, l in zip(RIGHT_CAL_DEG, LEFT_CAL_DEG)]
WHEEL_CAL_RAD = np.deg2rad(WHEEL_CAL_DEG).tolist()

LEFT_LIMIT_RAD  = np.deg2rad(-10.0)
RIGHT_LIMIT_RAD = np.deg2rad(18.75)
LEFT_SW_LIMIT   = LEFT_LIMIT_RAD
RIGHT_SW_LIMIT  = np.deg2rad(18.75)


# ------------------------------------------------------------------------------
#  Circular weighted mean — safe across ±π wrap
# ------------------------------------------------------------------------------
def fuse_angles(a1: float, w1: float, a2: float, w2: float) -> float:
    x = w1 * np.cos(a1) + w2 * np.cos(a2)
    y = w1 * np.sin(a1) + w2 * np.sin(a2)
    return float(np.arctan2(y, x))


class TwistTestNode(Node):
    def __init__(self):
        super().__init__("twist_test")

        # --- Parameters ---
        self.declare_parameter("wheelbase",         0.21)
        self.declare_parameter("wheel_diameter",    0.055)
        self.declare_parameter("rpm_filter_window", 5)
        self.declare_parameter("erpm_per_ms",       4550.0)
        self.declare_parameter("v_max_ms",          1.0)

        # EKF position blend factor:
        #   0.0 = ignore EKF position entirely (pure dead-reckoning)
        #   1.0 = snap to EKF position every cycle (no dead-reckoning)
        #   0.3 = recommended — gentle correction that kills long-term drift
        #         without causing jumps on EKF glitches
        self.declare_parameter("ekf_pos_blend",     0.3)

        self.wheelbase     = self.get_parameter("wheelbase").value
        self.wheel_radius  = self.get_parameter("wheel_diameter").value / 2.0
        filter_window      = self.get_parameter("rpm_filter_window").value
        self.erpm_per_ms   = self.get_parameter("erpm_per_ms").value
        self.v_max_ms      = self.get_parameter("v_max_ms").value
        self.ekf_pos_blend = self.get_parameter("ekf_pos_blend").value

        # --- State ---
        self.speed      = Float64()
        self.steer      = Float64()
        self.speed.data = 0.0
        self.steer.data = 0.0

        self.prev_theta         = 0.0
        self.prev_vel           = 0.0
        self.actual_vel         = 0.0
        self.curr_s_pos         = 0.53
        self.actual_wheel_angle = 0.0

        # RPM moving-average filter
        self._rpm_buf = deque(maxlen=filter_window)

        # Heading estimators — init to +90° (facing +Y)
        self.theta_   = np.deg2rad(90.0)   # fused heading
        self.count_ = 0
        # Position
        self.xpos = -1.6
        self.ypos = -1.4

        # --------------------------------------------------------------
        #  AprilTag IPS state
        #  last_tag_time: used to detect stale tag data — if the tag
        #  hasn't been seen for > TAG_TIMEOUT seconds we stop trusting it
        #  and fall back to EKF-blended dead-reckoning.
        # --------------------------------------------------------------
        self.TAG_TIMEOUT   = 1.0    # seconds — tune based on your IPS rate
        self.TAG_UPDATE_INTERVAL = 0.0 # accept a new tag reading only every 2 s


        self.last_tag_update_time = None
        self.last_tag_time = None
        self.tag_pos_valid = False

        self.quat           = quaternion_from_euler(0, 0, self.theta_)
        self.last_odom_time = None

        # --- Subscribers ---
        self.twist_sub_    = self.create_subscription(Twist,"/test_twist",self.callback_read,10)
        self.encoder_sub_  = self.create_subscription(VescStateStamped, "/sensors/core",self.encoder_callback,10)
        self.ekf_sub_      = self.create_subscription(Odometry,"/odometry/filtered", self.ekf_callback,10)

        # AprilTag IPS — hard overwrites position when tag is visible
        self.apriltag_sub_ = self.create_subscription(PoseStamped,"/apriltag/tag2/pose",self.apriltag_callback, 10)

        # --- Publishers ---
        self.speed_pub      = self.create_publisher(Float64,  "/commands/motor/speed",      10)
        self.steer_pub      = self.create_publisher(Float64,  "/commands/servo/position",   10)
        self.duty_cycle_pub = self.create_publisher(Float64,  "/commands/motor/duty_cycle", 10)
        self.odom_publisher = self.create_publisher(Odometry, "/odom/imu",                  10)
        self.marker_pub = self.create_publisher(Marker, "/car_bounds_marker", 10)

        # Debug topics
        self.debug_speed_pub  = self.create_publisher(Float64, "/debug/wheel_speed_ms",  10)
        self.debug_steer_pub  = self.create_publisher(Float64, "/debug/wheel_angle_rad", 10)
        self.debug_cmd_pub    = self.create_publisher(Float64, "/debug/cmd_speed_ms",    10)
        self.debug_theta1_pub = self.create_publisher(Float64, "/debug/theta1_deg",      10)
        self.debug_theta2_pub = self.create_publisher(Float64, "/debug/theta2_deg",      10)
        self.debug_tag_pub    = self.create_publisher(Float64, "/debug/tag_age_sec",     10)

        # --- Odom message template ---
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id  = "base_footprint_imu"
        self.odom_msg.pose.pose.position.x    = self.xpos
        self.odom_msg.pose.pose.position.y    = self.ypos
        self.odom_msg.pose.pose.position.z    = 0.0
        self.odom_msg.pose.pose.orientation.x = self.quat[0]
        self.odom_msg.pose.pose.orientation.y = self.quat[1]
        self.odom_msg.pose.pose.orientation.z = self.quat[2]
        self.odom_msg.pose.pose.orientation.w = self.quat[3]

        self.create_timer(0.02, self.update_odom)

        self.get_logger().info(
            f"TwistTestNode ready | wheel_radius={self.wheel_radius:.4f} m "
            f"| v_max={self.v_max_ms} m/s | erpm_per_ms={self.erpm_per_ms} "
            f"| ekf_pos_blend={self.ekf_pos_blend} "
            f"| initial heading=+90° (+Y) "
            f"| steer_limits: left={np.rad2deg(LEFT_SW_LIMIT):.1f}° "
            f"right={np.rad2deg(RIGHT_SW_LIMIT):.1f}°"
        )

    # ------------------------------------------------------------------
    #  Steering calibration lookup
    # ------------------------------------------------------------------
    def servo_to_wheel_angle(self, servo_pos: float) -> float:
        return float(np.interp(servo_pos, SERVO_CAL_POS, WHEEL_CAL_RAD))

    def angle_to_servo(self, angle_rad: float) -> float:
        SERVO_CENTER = 0.53
        if angle_rad <= 0.0:
            return float(np.interp(angle_rad,
                                   [LEFT_LIMIT_RAD, 0.0],
                                   [0.20, SERVO_CENTER]))
        else:
            return float(np.interp(angle_rad,
                                   [0.0, RIGHT_LIMIT_RAD],
                                   [SERVO_CENTER, 0.90]))

    # ------------------------------------------------------------------
    #  VESC callback → m/s
    # ------------------------------------------------------------------
    def encoder_callback(self, msg: VescStateStamped):
        erpm    = msg.state.speed
        raw_vel = erpm / 4395.0

        # if abs(self.prev_vel) < 0.05 and abs(raw_vel) > 0.3:
        #     self.get_logger().warn("Encoder spike suppressed (motor idle)")
        #     raw_vel = 0.0

        self._rpm_buf.append(raw_vel)
        self.actual_vel = float(np.mean(self._rpm_buf))

        debug_msg      = Float64()
        debug_msg.data = self.actual_vel
        self.debug_speed_pub.publish(debug_msg)

    # ------------------------------------------------------------------
    #  EKF callback → heading + gentle position correction
    #
    #  Heading  : extracted directly from quaternion (no re-integration)
    #  Position : blended toward EKF at ekf_pos_blend rate to kill
    #             long-term dead-reckoning drift without hard jumps.
    #             Skipped when AprilTag IPS is actively overriding position.
    # ------------------------------------------------------------------
    def ekf_callback(self, msg: Odometry):
        # Heading — extract yaw from quaternion directly
        q    = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta_ = float(np.arctan2(siny, cosy))
    # ------------------------------------------------------------------
    #  AprilTag IPS callback → hard position + heading reset
    #
    #  This is the highest-trust source. When tag1 is visible:
    #    - xpos/ypos are overwritten directly (no blending)
    #
    #  tag_pos_valid expires after TAG_TIMEOUT seconds in update_odom,
    #  after which the system falls back to EKF blending automatically.
    # ------------------------------------------------------------------
    def apriltag_callback(self, msg: PoseStamped):

        now = self.get_clock().now().nanoseconds * 1e-9
        # if (self.last_tag_update_time is not None and (now - self.last_tag_update_time) < self.TAG_UPDATE_INTERVAL):
        #     return

        self.last_tag_update_time = now
        # Hard overwrite position
        self.xpos = msg.pose.position.x
        self.ypos = msg.pose.position.y

        # Extract heading from tag and reset kinematic model
        q       = msg.pose.orientation
        siny    = 2.0 * (q.w * q.z + q.x * q.y)
        cosy    = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        tag_yaw = float(np.arctan2(siny, cosy))

        self.theta_ = tag_yaw

        # Mark tag as fresh
        self.last_tag_time = self.get_clock().now().nanoseconds * 1e-9
        self.tag_pos_valid = True

        # self.get_logger().info(f"IPS reset → x={self.xpos:.3f} y={self.ypos:.3f} "f"yaw={np.rad2deg(tag_yaw):.1f}°")

    # ------------------------------------------------------------------
    #  Twist command callback → motor speed + servo position
    # ------------------------------------------------------------------
    def callback_read(self, msg: Twist):
        dt             = 0.02
        MAX_STEER_RATE = 0.9
        MAX_ACCEL      = 2.0

        v_cmd_ms = float(np.clip(msg.linear.x, -self.v_max_ms, self.v_max_ms))

        dbg_cmd      = Float64()
        dbg_cmd.data = v_cmd_ms
        self.debug_cmd_pub.publish(dbg_cmd)

        vel_error     = np.clip(v_cmd_ms - self.prev_vel, -MAX_ACCEL * dt, MAX_ACCEL * dt)
        vel_final     = self.prev_vel + vel_error
        self.prev_vel = vel_final

        speed_erpm = float(vel_final * self.erpm_per_ms)

        s_cmd         = -msg.angular.z    # negated to match physical linkage
        delta_desired = 0.0 if abs(v_cmd_ms) < 0.05 else s_cmd
        delta_desired = float(np.clip(delta_desired, LEFT_SW_LIMIT, RIGHT_SW_LIMIT))

        max_step    = MAX_STEER_RATE * dt
        theta_error = np.clip(delta_desired - self.prev_theta, -max_step, max_step)
        theta_final = self.prev_theta + theta_error
        self.prev_theta = theta_final

        target_servo    = self.angle_to_servo(theta_final)
        self.curr_s_pos += (target_servo - self.curr_s_pos) * 0.52
        self.curr_s_pos  = float(np.clip(self.curr_s_pos, 0.20, 0.90))

        self.actual_wheel_angle = self.servo_to_wheel_angle(self.curr_s_pos)

        dbg_steer      = Float64()
        dbg_steer.data = self.actual_wheel_angle
        self.debug_steer_pub.publish(dbg_steer)

        self.speed.data = speed_erpm
        self.steer.data = self.curr_s_pos
        self.speed_pub.publish(self.speed)
        self.steer_pub.publish(self.steer)

    # ------------------------------------------------------------------
    #  Visualization Marker (21x17 cm box)
    # ------------------------------------------------------------------
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
        marker.color.a = 1.0  # Alpha (transparency)

        self.marker_pub.publish(marker)

    # ------------------------------------------------------------------
    #  Odometry update (50 Hz timer)
    # ------------------------------------------------------------------
    def update_odom(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_odom_time is None:
            self.last_odom_time = now
            return
        dt = now - self.last_odom_time
        self.last_odom_time = now

        if dt <= 0.0:
            return

        # --------------------------------------------------------------
        #  Check tag freshness — expire after TAG_TIMEOUT seconds.
        #  Once expired, EKF position blending resumes automatically.
        # --------------------------------------------------------------
        if self.last_tag_time is not None:
            tag_age = now - self.last_tag_time
            if tag_age > self.TAG_TIMEOUT:
                if self.tag_pos_valid:
                    self.get_logger().warn(
                        f"IPS tag lost ({tag_age:.2f}s ago) — falling back to EKF blend"
                    )
                self.tag_pos_valid = False

            tag_age_msg      = Float64()
            tag_age_msg.data = float(tag_age)
            self.debug_tag_pub.publish(tag_age_msg)

        # Dead-reckoning position
        self.xpos += self.actual_vel * np.cos(self.theta_) * dt
        self.ypos += self.actual_vel * np.sin(self.theta_) * dt

        # Fused heading — EKF weight 4, kinematic weight 1
        self.theta_ = self.theta_

        self.quat = quaternion_from_euler(0, 0, self.theta_)

        self.odom_msg.header.stamp             = self.get_clock().now().to_msg()
        self.odom_msg.pose.pose.position.x     = self.xpos
        self.odom_msg.pose.pose.position.y     = self.ypos
        self.odom_msg.pose.pose.orientation.x  = self.quat[0]
        self.odom_msg.pose.pose.orientation.y  = self.quat[1]
        self.odom_msg.pose.pose.orientation.z  = self.quat[2]
        self.odom_msg.pose.pose.orientation.w  = self.quat[3]
        self.odom_msg.twist.twist.linear.x     = self.actual_vel
        self.odom_msg.twist.twist.angular.z    = ((self.actual_vel / self.wheelbase) * np.tan(self.actual_wheel_angle))

        self.odom_publisher.publish(self.odom_msg)
        self.publish_car_box_marker()


def main(args=None):
    rclpy.init(args=args)
    node = TwistTestNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()