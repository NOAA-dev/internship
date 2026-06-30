#!/usr/bin/env python3
"""
mpc_node.py
===========

ROS 2 node implementing a Linear Time-Varying Model Predictive Controller
(LTV-MPC) for an Ackermann-steered ground robot (F1Tenth-style platform).

At every odometry update the controller:
  1. Linearizes the kinematic bicycle model around the current state and
     the previously-computed control sequence (one linearization per
     horizon step -> "LTV").
  2. Builds the condensed (state-free) quadratic program for the whole
     horizon using the standard batch formulation:

         X = Phi @ x0 + Gamma @ U + C

     where X stacks the predicted states, U stacks the predicted
     controls, and Phi/Gamma/C are built by propagating the per-step
     linearizations A_k, B_k, g_k.
  3. Solves the QP with OSQP (warm-started from the previous solution for
     fast convergence) subject to:
       - absolute actuator limits (steering angle, acceleration),
       - actuator rate limits (steering slew rate, jerk),
       - velocity bounds, expressed as linear constraints on U through
         the condensed model.
  4. Applies the first control input of the solved sequence and shifts
     the rest forward as the warm start / nominal trajectory for the
     next cycle.

Beyond core tracking, the node also owns:
  * Reference-horizon generation from a globally-planned path, including
    speed tapering near the goal and through curvature, and detection of
    path segments that should be driven in reverse.
  * A lightweight obstacle-avoidance fallback: if the MPC-predicted
    trajectory runs into an obstacle (checked against a precomputed
    Euclidean distance-transform map), the node drops out of MPC and
    runs an open-loop "escape" maneuver (forward or reverse, whichever
    direction has more clearance) for a fixed dwell time before
    resuming closed-loop tracking.
  * Position-jump detection (e.g. triggered by a vision-based localizer
    re-locking) that requests a Hybrid A* replan and holds the robot
    stationary until a new path arrives.
  * Reacting to perception-driven traffic-sign commands (STOP /
    SLOW_DOWN / PARK) published as plain strings by a separate sign
    detection node.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import rclpy
import osqp
import numpy as np
from scipy import sparse, ndimage

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf_transformations import euler_from_quaternion

from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from visualization_msgs.msg import Marker

# Custom message/service types.
# NOTE: `PathMsg` (custom_interfaces) is the *input* global path -- it carries
# raw x/y/theta arrays from the planner. `Path` (nav_msgs) is the *output*
# message type used for all the visualization/debug topics published below.
from custom_interfaces.msg import Path as PathMsg
from custom_interfaces.srv import HybridAStar


class MPCNode(Node):
    """LTV-MPC trajectory-tracking controller for the autonomous vehicle.

    Subscribes to the globally-planned path and filtered odometry, and
    publishes the velocity/steering command together with several
    `nav_msgs/Path` topics used for visualization and debugging (predicted
    horizon, reference horizon, open-loop nominal rollout, executed path).
    """

    def __init__(self):
        super().__init__("mpc_node")

        # --- Discretization & vehicle model ---------------------------------
        self.dt_ = 0.1                          # control/prediction time step [s]
        self.L_ = 0.21                          # wheelbase [m]
        self.state_ = np.array([0.0, 0.0, 1.57, 0.0])  # [x, y, theta, v]
        self.N_ = 25                            # horizon length [steps]

        # control_input_ holds the *nominal* control sequence used for
        # linearization (steering angle, acceleration) for each horizon
        # step; it is shifted forward each cycle using the previous
        # solution as a warm-start trajectory.
        self.control_input_ = np.zeros((self.N_, 2))
        # Last applied (k=0) control input -- used to enforce rate limits
        # on the very first step of the next horizon.
        self.prev_optimal_control_input_ = np.array([0.0, 0.0])
        self.prev_solution_ = None              # OSQP warm-start vector
        self.vel_ = 0.0                         # commanded velocity sent downstream

        # --- Cost weights (Bryson's rule tuned) ------------------------------
        # State weights: [x, y, theta, v]. Terminal step uses Q_ * 10.
        self.Q_ = np.array([[60, 0, 0, 0],
                             [0, 60, 0, 0],
                             [0, 0, 80, 0],
                             [0, 0, 0, 10]])
        # Control weights: [steering angle, acceleration].
        self.R_ = np.array([[30, 0],
                             [0, 25]])

        # --- Actuator & state limits -----------------------------------------
        # Asymmetric steering limits measured on the physical hardware:
        # "Left" maps to the negative bound, "Right" to the positive bound.
        self.max_steering_angle_Left = -np.radians(18.75)
        self.max_steering_angle_Right = np.radians(10.0)
        self.max_acceleration_ = 1.0
        self.min_acceleration_ = -1.5
        self.max_velocity_ = 0.62
        self.min_velocity_ = -0.4
        self.max_steering_change_ = np.radians(7)   # per-step steering slew limit
        self.max_acceleration_change_ = 0.35        # per-step jerk limit
        self.SLOWDOWN_RADIUS_ = 0.5                  # start tapering speed within this radius of goal
        self.GOAL_VEL_ = 0.2                         # commanded speed right at the goal

        # --- Static occupancy map & distance-transform (for collision checks) ---
        self.raw_map = np.load(
            "/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/occupancy_grid_mpc.npy"
        )
        # Euclidean distance transform: dist_field_[row, col] = distance (m)
        # from that cell to the nearest occupied cell.
        self.dist_field_ = ndimage.distance_transform_edt(1 - self.raw_map, sampling=0.01)
        self.map_shape_ = self.dist_field_.shape

        # --- Obstacle-avoidance / escape-maneuver tuning ---------------------
        self.slow_radius_ = 0.15        # clearance below which we slow down
        self.stop_radius_ = 0.11        # clearance below which we trigger an escape
        self.probe_distance_ = 0.12     # forward/backward probe distance used to pick escape direction
        self.reverse_speed_ = -0.25
        self.forward_speed_ = 0.17
        self.escape_direction_ = None   # +1 = committed forward, -1 = committed reverse, None = not escaping
        self.escape_until_time_ = None  # rclpy Time -- hold escape_direction_ until this passes
        self.ESCAPE_DWELL_SEC_ = 1.0    # minimum time to commit to one direction once chosen
        self.escape_steer_ = 0.0
        self._last_escape_steer_ = 0.0

        # --- Map <-> world coordinate transform ------------------------------
        self.map_resolution_ = 0.01
        self.origin_x_ = -3.0
        self.origin_y_ = -2.5

        # --- ROS interfaces ---------------------------------------------------
        # Global path is published TRANSIENT_LOCAL so a late-joining MPC node
        # still receives the last published path.
        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.path_sub_ = self.create_subscription(PathMsg, "/path", self.path_callback, path_qos)
        self.sub_odom_ = self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.msg_sub_ = self.create_subscription(String, "/sign_action_node/command", self.listen_msg, 10)

        self.control_pub_ = self.create_publisher(Float64MultiArray, "/velocity_steer", 10)
        self.path_trace_pub_ = self.create_publisher(Path, "/robot_actual_path", 10)
        self.pred_path_pub_ = self.create_publisher(Path, "/mpc_predicted_path", 10)
        self.obs_pub_ = self.create_publisher(Marker, "/mpc_nearby_obstacles", 10)
        self.reference_path_pub_ = self.create_publisher(Path, "/mpc_reference_path", 10)
        self.nominal_path_pub_ = self.create_publisher(Path, "/mpc_nominal_path", 10)
        self.twist_pub_ = self.create_publisher(Twist, "/test_twist", 10)

        self.path_trace_ = Path()
        self.path_trace_.header.frame_id = "map"

        # Global path arrays (filled in by path_callback).
        self.path_x_ = []
        self.path_y_ = []
        self.path_theta_ = []

        # --- Startup ramp-up state --------------------------------------------
        self.startup_complete_ = False
        self.STARTUP_VEL_THRESHOLD = 0.1

        # --- Hybrid A* replan client & position-jump detector -----------------
        self.replan_client_ = self.create_client(HybridAStar, "/goal")

        self.prev_pos_ = None            # last known good position
        self.JUMP_THRESHOLD_ = 0.5       # metres -- tune between 0.70-0.80
        self.waiting_replan_ = False     # True = stop and wait for new path
        self.goal_ = None                # stores current goal for replan request
        self.goal_reached_ = False
        self._last_idx_ = None
        self.startup_time_ = None

        # --- Sign-reaction (STOP / SLOW_DOWN / PARK) state ---------------------
        self.last_msg_time_ = self.get_clock().now()
        self.wait_time_ = self.get_clock().now()
        self.active_command = None
        self.command_start_time = None
        self.msg = None
        self.stop_serviced_ = False
        self.none_count_ = 0
        self.NONE_THRESH_ = 20

    # -------------------------------------------------------------------------
    # Position-jump detection & Hybrid A* replanning
    # -------------------------------------------------------------------------
    def check_position_jump(self):
        """Flag a discontinuity between consecutive odometry readings.

        Used to catch large corrections from the localizer (e.g. an
        AprilTag-based re-lock) that invalidate the current MPC tracking
        index and warrant a fresh Hybrid A* plan rather than trying to
        track straight through the jump.

        Returns:
            bool: True if the position moved more than `JUMP_THRESHOLD_`
            metres since the last call, False otherwise. Always updates
            `prev_pos_` to the current position as a side effect.
        """
        if self.prev_pos_ is None:
            self.prev_pos_ = self.state_[0:2].copy()
            return False

        jump = np.linalg.norm(self.state_[0:2] - self.prev_pos_)

        if jump > self.JUMP_THRESHOLD_:
            self.get_logger().warn(
                f"Position jump detected: {jump:.3f} m — requesting replan"
            )
            self.prev_pos_ = self.state_[0:2].copy()
            return True

        self.prev_pos_ = self.state_[0:2].copy()
        return False

    def _replan_retry_once(self):
        """One-shot timer callback: retry a previously failed replan request."""
        self.request_replan()

    def request_replan(self):
        """Send an asynchronous Hybrid A* replan request for the current goal.

        No-ops (with a warning) if no goal has been set yet or the
        planning service is not currently available.
        """
        if self.goal_ is None:
            self.get_logger().warn("Cannot replan — no goal set")
            return

        if not self.replan_client_.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Hybrid A* service not available")
            return

        req = HybridAStar.Request()
        req.goal = self.goal_  # float64[3]: x, y, theta of original goal

        future = self.replan_client_.call_async(req)
        future.add_done_callback(self.replan_response_callback)

    def replan_response_callback(self, future):
        """Handle the result of an asynchronous Hybrid A* replan request.

        On success, clears all motion state that should not survive a
        replan (warm start, startup ramp, commanded velocity) so the
        controller resumes cleanly on the new path. On failure, schedules
        a retry.

        NOTE: `self._retry_timer_` is referenced here (`.cancel()` /
        `.destroy()`) but the retry branch below only does
        `self.create_timer(...)` without storing the handle as
        `self._retry_timer_`. As written, a successful replan that
        follows a prior failed attempt will raise an AttributeError on
        this line. Left unchanged here since this rewrite is documentation
        -only, but flagging it for a follow-up fix.
        """
        try:
            res = future.result()
            if "successful" in res.feedback:
                self.get_logger().info("Replan successful — resuming MPC")
                self.goal_reached_ = False
                self.waiting_replan_ = False
                self.startup_complete_ = False   # ramp velocity up again from rest
                self.vel_ = 0.0
                self.prev_solution_ = None       # clear stale OSQP warm start
                self._replan_retry_once.cancel()
                self._replan_retry_once.destroy()
            else:
                self.get_logger().warn(f"Replan failed: {res.feedback} — retrying")
                # retry after 20 second
                self.create_timer(5.0, lambda: self._replan_retry_once())
        except Exception as e:
            self.get_logger().error(f"Replan service call failed: {e}")
            self.waiting_replan_ = False

    # -------------------------------------------------------------------------
    # Global path ingestion
    # -------------------------------------------------------------------------
    def path_callback(self, msg):
        """Handle an incoming global path from the planner.

        Two cases are distinguished by comparing the incoming goal to the
        currently stored goal:

          * Same goal (within 5 cm): the planner has republished a refreshed
            version of the path to the *same* destination (e.g. after a
            local replan around a transient obstacle). If the geometry is
            actually unchanged, this is a no-op. Otherwise a "soft update"
            replaces the path arrays without resetting velocity, the
            startup ramp, or the OSQP warm start (`_soft_update_path`).

          * New goal: full reset of all path- and motion-related state,
            followed by a check of whether the first path segment points
            opposite to the initial heading (`reverse_start_`), which tells
            `mpc_control` to ramp velocity downward instead of upward
            during the startup phase.

        Args:
            msg: PathMsg with parallel arrays `x`, `y`, `theta` describing
                the planned path; the path heading is smoothed with a
                5-point moving average after unwrapping.
        """
        # 1. Extract the incoming goal from the message
        incoming_goal = [msg.x[-1], msg.y[-1], msg.theta[-1]]

        new_x = np.array(msg.x)
        new_y = np.array(msg.y)
        theta = np.unwrap(np.array(msg.theta))
        kernel = np.ones(5) / 5
        new_theta = np.convolve(np.pad(theta, 2, mode='edge'), kernel, mode='valid')

        same_goal = False
        if self.goal_ is not None:
            goal_dist = np.hypot(incoming_goal[0] - self.goal_[0],
                                  incoming_goal[1] - self.goal_[1])
            same_goal = goal_dist < 0.05

        if same_goal:
            # Same destination -- this is a refreshed/continuously-republished
            # path for the SAME goal, not a brand new task. Check if the
            # geometry actually changed before doing any work.
            if (len(new_x) == len(self.path_x_) and
                    np.allclose(new_x, self.path_x_, atol=1e-4) and
                    np.allclose(new_y, self.path_y_, atol=1e-4)):
                return  # identical path, nothing to do

            self.get_logger().info("Path geometry updated for current goal -- soft update")
            self._soft_update_path(new_x, new_y, new_theta)
            return

        # 2. Truly new goal -- full reset (this is the old "brand new path" branch)
        self.path_x_ = new_x
        self.path_y_ = new_y
        self.path_theta_ = new_theta
        self.goal_ = incoming_goal

        self.goal_reached_ = False
        self.waiting_replan_ = False
        self.startup_complete_ = False
        self.prev_solution_ = None
        self._last_idx_ = None
        self.startup_time_ = None
        self.reverse_start_ = False

        self.get_logger().info(f"Successfully received a BRAND NEW path. Goal: {self.goal_}")
        self.control_input_ = np.zeros((self.N_, 2))
        self.prev_optimal_control_input_ = np.array([0.0, 0.0])

        self._rebuild_arc_length()

        if len(self.path_x_) > 1:
            dx = self.path_x_[1] - self.path_x_[0]
            dy = self.path_y_[1] - self.path_y_[0]
            segment_heading = np.arctan2(dy, dx)
            first_heading = self.path_theta_[0]
            heading_diff = abs(np.arctan2(np.sin(segment_heading - first_heading),
                                           np.cos(segment_heading - first_heading)))
            self.reverse_start_ = heading_diff > np.pi / 2

    def _rebuild_arc_length(self):
        """Recompute the cumulative arc-length array `path_s_` for `path_x_`/`path_y_`.

        `path_s_[i]` is the distance travelled along the path from the
        first point to point `i`. This lets the reference-horizon
        generator look ahead by *distance* rather than by index, which is
        robust to non-uniform point spacing.
        """
        self.path_s_ = [0.0]
        for i in range(1, len(self.path_x_)):
            dx = self.path_x_[i] - self.path_x_[i - 1]
            dy = self.path_y_[i] - self.path_y_[i - 1]
            self.path_s_.append(self.path_s_[-1] + np.hypot(dx, dy))
        self.path_s_ = np.array(self.path_s_)

        dist = np.diff(self.path_s_)
        if len(dist) > 0:
            self.get_logger().info(f"Average path spacing = {np.mean(dist):.3f} m")

    def _soft_update_path(self, new_x, new_y, new_theta):
        """Replace path geometry for the SAME goal without resetting motion state.

        Preserves velocity, the startup ramp, and the OSQP warm start, but
        re-anchors `_last_idx_` against the new array by doing a full
        nearest-point search (not the forward-only search used in
        steady-state tracking) -- the new path's indexing has no relation
        to the old one, so a forward-only search could get stuck or jump
        backward incorrectly.

        Args:
            new_x, new_y, new_theta: replacement path arrays.
        """
        old_idx = self._last_idx_

        self.path_x_ = new_x
        self.path_y_ = new_y
        self.path_theta_ = new_theta
        self._rebuild_arc_length()

        # Re-anchor the tracking index: find the closest point on the NEW
        # path to the robot's current position (full search, not forward-only,
        # since the new path's indexing has no relation to the old one).
        if len(self.path_x_) > 0:
            dx = self.path_x_ - self.state_[0]
            dy = self.path_y_ - self.state_[1]
            dist = dx * dx + dy * dy
            self._last_idx_ = int(np.argmin(dist))
        else:
            self._last_idx_ = None

        self.get_logger().info(
            f"Soft path update: re-anchored index {old_idx} -> {self._last_idx_}"
        )

    def find_closest_path_index(self):
        """Find the path index closest to the robot, searching forward only.

        Starting the search at `_last_idx_` (rather than over the whole
        path) keeps this O(remaining path) instead of O(path) per cycle,
        and guarantees the tracking index is monotonically non-decreasing,
        which prevents the reference horizon from snapping backward on a
        path that loops near itself. A small forward bias additionally
        nudges the index ahead by one point if the robot has already
        passed the nominally-closest point, to avoid lingering exactly at
        the closest point.

        Returns:
            int: index into `path_x_`/`path_y_`/`path_theta_` of the
            (forward-biased) closest point. Also stored in `_last_idx_`.
        """
        dx = self.path_x_ - self.state_[0]
        dy = self.path_y_ - self.state_[1]
        dist = dx * dx + dy * dy

        # Only search from last known index forward — never go backward
        if self._last_idx_ is not None:
            search_start = self._last_idx_   # no lookback — forward only
            dist_forward = dist[search_start:]
            raw_idx = search_start + int(np.argmin(dist_forward))
        else:
            raw_idx = int(np.argmin(dist))

        # Forward bias
        if raw_idx < len(self.path_x_) - 1:
            path_dir = np.array([
                self.path_x_[raw_idx + 1] - self.path_x_[raw_idx],
                self.path_y_[raw_idx + 1] - self.path_y_[raw_idx]
            ])
            to_robot = np.array([
                self.state_[0] - self.path_x_[raw_idx],
                self.state_[1] - self.path_y_[raw_idx]
            ])
            if np.dot(path_dir, to_robot) > 0:
                raw_idx = min(raw_idx + 1, len(self.path_x_) - 1)

        self._last_idx_ = raw_idx
        return raw_idx

    # -------------------------------------------------------------------------
    # Reference horizon construction
    # -------------------------------------------------------------------------
    def build_reference_horizon(self):
        """Build the [x, y, theta, v] reference trajectory tracked by the MPC.

        The reference speed `v_cmd` is the minimum of three independent
        caps, each addressing a different reason to slow down:
          1. **Goal slowdown** -- linear taper from cruise speed down to
             `GOAL_VEL_` within `SLOWDOWN_RADIUS_` of the goal.
          2. **Curvature lookahead** -- if the heading change over a short
             lookahead distance exceeds `TURN_THRESHOLD_DEG`, cap speed
             down towards `TURN_VEL` proportionally to how sharp the turn
             is, so the vehicle slows down *before* entering a turn.
          3. **Remaining-arc-length taper** -- an additional taper that
             kicks in once less than 0.4 m of path remains, independent of
             straight-line distance to the goal (handles paths that curl
             back near the goal).

        The per-step lookahead distance along the path is computed using a
        velocity estimate `v_ref` (clamped to a sane minimum so the horizon
        does not collapse to a single point at low/zero speed), which is
        itself shrunk near the end of the path so the last few horizon
        steps don't shoot past the goal.

        Path segments whose direction disagrees with the stored heading by
        more than 90 degrees are treated as reverse segments: the
        reference velocity for that step is negated.

        Once the lookahead runs past the end of the path ("path
        exhausted"), all remaining horizon steps are frozen at the goal
        pose with zero reference velocity.

        Returns:
            np.ndarray: shape (N_, 4) reference trajectory, one row per
            horizon step, columns [x, y, theta, v].
        """
        idx0 = self.find_closest_path_index()
        s0 = self.path_s_[idx0]

        # Distance to goal for velocity taper
        dist_to_goal = np.linalg.norm(
            self.state_[0:2] - np.array([self.path_x_[-1], self.path_y_[-1]])
        )

        # --- Goal slowdown ---
        if dist_to_goal < self.SLOWDOWN_RADIUS_:
            t = dist_to_goal / self.SLOWDOWN_RADIUS_
            v_cmd = self.GOAL_VEL_ + t * (0.52 - self.GOAL_VEL_)
        else:
            v_cmd = 0.52

        # --- Curvature lookahead slowdown ---
        LOOKAHEAD_S = 0.4
        s_ahead = s0 + LOOKAHEAD_S
        idx_ahead = int(np.searchsorted(self.path_s_, s_ahead))
        idx_ahead = min(idx_ahead, len(self.path_x_) - 1)

        heading_now = self.path_theta_[idx0]
        heading_ahead = self.path_theta_[idx_ahead]
        curvature_deg = abs(np.degrees(np.arctan2(
            np.sin(heading_ahead - heading_now),
            np.cos(heading_ahead - heading_now)
        )))

        TURN_THRESHOLD_DEG = 5.0
        TURN_VEL = 0.25
        if curvature_deg > TURN_THRESHOLD_DEG:
            t_turn = min((curvature_deg - TURN_THRESHOLD_DEG) / 12.0, 1.0)
            v_cmd = min(v_cmd, TURN_VEL + (1.0 - t_turn) * (0.52 - TURN_VEL))

        # --- Remaining arc length taper ---
        remaining_s = self.path_s_[-1] - s0
        if remaining_s < 0.4:
            taper = max(remaining_s / 0.4, 0.0)
            v_cmd = min(v_cmd, self.GOAL_VEL_ + taper * (0.52 - self.GOAL_VEL_))

        # Shrink lookahead when near end so horizon doesn't collapse
        v_ref = max(abs(self.state_[3]), 0.35)
        if remaining_s < 0.4:
            v_ref = max(v_ref * max(remaining_s / 0.4, 0.3), 0.15)

        ref_traj = np.zeros((self.N_, 4))
        path_exhausted = False
        freeze_x = 0.0
        freeze_y = 0.0
        freeze_theta = 0.0

        for k in range(self.N_):
            s_ref = s0 + v_ref * (k + 1) * self.dt_
            idx = int(np.searchsorted(self.path_s_, s_ref))
            idx = min(idx, len(self.path_x_) - 1)

            if idx >= len(self.path_x_) - 1 and not path_exhausted:
                path_exhausted = True
                freeze_x = self.path_x_[-1]
                freeze_y = self.path_y_[-1]
                freeze_theta = float(self.goal_[2])

            if path_exhausted:
                ref_traj[k, 0] = freeze_x
                ref_traj[k, 1] = freeze_y
                ref_traj[k, 2] = freeze_theta
                ref_traj[k, 3] = 0.0
                continue

            ref_traj[k, 0] = self.path_x_[idx]
            ref_traj[k, 1] = self.path_y_[idx]
            ref_traj[k, 3] = v_cmd

            if idx < len(self.path_x_) - 1:
                dx = self.path_x_[idx + 1] - self.path_x_[idx]
                dy = self.path_y_[idx + 1] - self.path_y_[idx]
                segment_heading = np.arctan2(dy, dx)
                path_theta = self.path_theta_[idx]

                heading_diff = abs(np.arctan2(
                    np.sin(segment_heading - path_theta),
                    np.cos(segment_heading - path_theta)
                ))

                if heading_diff > np.pi / 2:
                    ref_traj[k, 3] = -v_cmd
                    ref_traj[k, 2] = path_theta
                else:
                    ref_traj[k, 2] = path_theta
            else:
                ref_traj[k, 2] = self.path_theta_[idx]

        return ref_traj

    def _compute_escape_steering(self, x, y, yaw, chosen_dir):
        """Proportional heading-alignment steering command for escape maneuvers.

        While the controller is in an open-loop "escape" (driving away from
        a predicted collision), this computes a simple P-controller steering
        command that tries to re-align the vehicle's heading with the
        direction back towards the path, looked ahead by
        `ALIGN_LOOKAHEAD_S` along the stored arc length from the last
        tracked index.

        When escaping in reverse (`chosen_dir < 0`), the reference heading
        is flipped by pi, since the vehicle is moving backward relative to
        its own heading.

        The output is both rate-limited (relative to the previous escape
        steering command) and saturated to the physical steering limits.

        Args:
            x, y, yaw: current vehicle pose.
            chosen_dir: +1 for forward escape, -1 for reverse escape.

        Returns:
            float: steering angle command [rad].
        """
        ALIGN_LOOKAHEAD_S = 0.3
        ESCAPE_STEER_KP = 1.2

        if self._last_idx_ is None or len(self.path_x_) == 0:
            return 0.0

        s_target = self.path_s_[self._last_idx_] + ALIGN_LOOKAHEAD_S
        idx_target = int(np.searchsorted(self.path_s_, s_target))
        idx_target = min(idx_target, len(self.path_x_) - 1)

        dx_path = self.path_x_[idx_target] - x
        dy_path = self.path_y_[idx_target] - y
        heading_to_path = np.arctan2(dy_path, dx_path)

        reference_yaw = yaw if chosen_dir > 0 else (yaw + np.pi)
        heading_error = np.arctan2(
            np.sin(heading_to_path - reference_yaw),
            np.cos(heading_to_path - reference_yaw)
        )

        steer_cmd = ESCAPE_STEER_KP * heading_error

        max_step = 0.9 * self.dt_
        steer_cmd = float(np.clip(steer_cmd,
                                   self._last_escape_steer_ - max_step,
                                   self._last_escape_steer_ + max_step))

        steer_cmd = float(np.clip(steer_cmd,
                                   self.max_steering_angle_Left,
                                   self.max_steering_angle_Right))

        self._last_escape_steer_ = steer_cmd
        return steer_cmd

    # -------------------------------------------------------------------------
    # QP cost & model assembly
    # -------------------------------------------------------------------------
    def generate_Q_and_R_matrices(self):
        """Assemble the block-diagonal horizon weighting matrices.

        Stacks `Q_` (state weight) `N_` times along the diagonal, with the
        final block scaled by 10 to weight terminal tracking error more
        heavily, and stacks `R_` (control weight) `N_` times along the
        diagonal.

        Returns:
            tuple[np.ndarray, np.ndarray]: (Q, R) of shapes
            (4*N_, 4*N_) and (2*N_, 2*N_) respectively.
        """
        Q = np.zeros((4 * self.N_, 4 * self.N_))
        R = np.zeros((2 * self.N_, 2 * self.N_))
        for i in range(self.N_):
            if i == self.N_ - 1:
                Q[4 * i:4 * (i + 1), 4 * i:4 * (i + 1)] = self.Q_ * 10
            else:
                Q[4 * i:4 * (i + 1), 4 * i:4 * (i + 1)] = self.Q_
            R[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = self.R_
        return Q, R

    def linearize_dynamics(self, state, control_input):
        """First-order (Euler) linearization of the bicycle model.

        Computes the continuous-time Jacobians `Ac = df/dstate`,
        `Bc = df/dcontrol` of the kinematic bicycle model at
        `(state, control_input)`, then discretizes them with a simple
        forward-Euler step (`A = I + Ac*dt`, `B = Bc*dt`) to match the
        discretization used by `kinematic_model`.

        Args:
            state: [x, y, theta, v] linearization point.
            control_input: [steering_angle, acceleration] linearization point.

        Returns:
            tuple[np.ndarray, np.ndarray]: discrete (A, B) matrices, shapes
            (4, 4) and (4, 2).
        """
        Ac = np.array([[0, 0, -state[3] * np.sin(state[2]), np.cos(state[2])],
                       [0, 0, state[3] * np.cos(state[2]), np.sin(state[2])],
                       [0, 0, 0, np.tan(control_input[0]) / self.L_],
                       [0, 0, 0, 0]])
        Bc = np.array([[0, 0],
                       [0, 0],
                       [(state[3] / self.L_) * (1 / np.cos(control_input[0]) ** 2), 0],
                       [0, 1]])

        A = np.eye(4) + Ac * self.dt_
        B = Bc * self.dt_
        return A, B

    def kinematic_model(self, state, control_input):
        """Nonlinear discrete-time kinematic bicycle model (forward Euler).

        Used both to simulate the open-loop nominal trajectory for
        linearization/warm-start purposes, and (implicitly, via
        `linearize_dynamics`) as the model the MPC is linearized around.

        Args:
            state: [x, y, theta, v].
            control_input: [steering_angle, acceleration].

        Returns:
            np.ndarray: next state [x, y, theta, v].
        """
        x_next = state[0] + state[3] * np.cos(state[2]) * self.dt_
        y_next = state[1] + state[3] * np.sin(state[2]) * self.dt_
        theta_next = state[2] + (state[3] / self.L_) * np.tan(control_input[0]) * self.dt_
        v_next = state[3] + control_input[1] * self.dt_

        return np.array([x_next, y_next, theta_next, v_next])

    # -------------------------------------------------------------------------
    # Static map lookups (for the obstacle-avoidance fallback)
    # -------------------------------------------------------------------------
    def world_to_pixel(self, x, y):
        """Convert world-frame coordinates to occupancy-grid (row, col).

        Args:
            x, y: world-frame coordinates [m].

        Returns:
            tuple[int, int]: (row, col) indices into `dist_field_`/`raw_map`.
        """
        col = int(round((x - self.origin_x_) / self.map_resolution_))
        row = int(round((y - self.origin_y_) / self.map_resolution_))

        return row, col

    def clearance_at(self, x, y):
        """Look up distance to the nearest obstacle at a world coordinate.

        Args:
            x, y: world-frame coordinates [m].

        Returns:
            float: clearance in metres from the precomputed distance
            transform, or 0.0 if the point falls outside the map bounds
            (treated as "no clearance" / unsafe).
        """
        row, col = self.world_to_pixel(x, y)
        rows, cols = self.map_shape_

        if row < 0 or row >= rows:
            return 0.0

        if col < 0 or col >= cols:
            return 0.0

        return float(self.dist_field_[row, col])

    def predicted_path_safe(self, X_pred):
        """Classify the safety of the MPC-predicted horizon against the map.

        Checks a fixed mid-horizon window (steps 5-8 inclusive) of the
        predicted trajectory -- close enough to be actionable, far enough
        ahead to give time to react -- against the clearance map.

        Args:
            X_pred: flat array of length 4*N_ (the condensed-form predicted
                states `Phi @ x0 + Gamma @ U + C`), reshaped internally to
                (N_, 4).

        Returns:
            tuple[str, int]: a status in {"STOP", "SLOW", "SAFE"} and the
            horizon index that triggered a STOP (or -1 if not STOP).
        """
        states = X_pred.reshape(self.N_, 4)
        minimum_clearance = 100.0

        for i in range(5, 9):
            x = states[i, 0]
            y = states[i, 1]
            c = self.clearance_at(x, y)
            minimum_clearance = min(minimum_clearance, c)

            if c < self.stop_radius_:
                self.get_logger().warn(f"Collision at step={i} "f"x={x:.3f} "f"y={y:.3f} "f"clearance={c:.3f}")
                return "STOP", i
        # if minimum_clearance < 100:
        #     # self.get_logger().info(f"min predicted clearance = {minimum_clearance:.3f}")
        if minimum_clearance < self.slow_radius_:
            return "SLOW", -1

        return "SAFE", -1

    # -------------------------------------------------------------------------
    # Sign-reaction state machine (STOP / SLOW_DOWN / PARK)
    # -------------------------------------------------------------------------
    def listen_msg(self, msg):
        """Update the active sign-reaction command from a perception string.

        STOP is debounced: it is only *triggered* once per detection
        episode (`stop_serviced_` latch), and is only cleared once "NONE"
        has been seen for `NONE_THRESH_` consecutive messages, so a single
        missed frame doesn't prematurely release the stop.

        Args:
            msg: std_msgs/String with data in {"STOP", "NONE", "SLOW_DOWN",
                "PARK"}.
        """
        now = self.get_clock().now()
        if msg.data == "STOP":
            # Sign visible again
            self.none_count_ = 0
            # Trigger stop only once
            if not self.stop_serviced_:
                self.active_command = "STOP"
                self.command_start_time = now
                self.stop_serviced_ = True

        elif msg.data == "NONE":
            # Count consecutive NONE detections
            self.none_count_ += 1
            # Sign considered truly gone
            if self.none_count_ >= self.NONE_THRESH_:
                self.stop_serviced_ = False

        elif msg.data == "SLOW_DOWN":
            self.active_command = "SLOW_DOWN"
            self.command_start_time = now

        elif msg.data == "PARK":
            self.active_command = "PARK"
            self.command_start_time = now

    def act_on_sign(self):
        """Apply the currently active sign-triggered behaviour to `vel_`.

        STOP fully overrides the MPC velocity to zero for 5 seconds and
        signals the caller to skip MPC entirely this cycle. SLOW_DOWN and
        PARK instead just clamp whatever velocity the MPC/startup ramp
        already produced, for 2 seconds, without overriding the MPC.

        Returns:
            bool: True if MPC should be bypassed this cycle (STOP active),
            False otherwise (no active command, or a clamp-only command).
        """
        if self.active_command is None:
            return False

        elapsed = (self.get_clock().now() - self.command_start_time).nanoseconds * 1e-9

        if self.active_command == "STOP":
            self.vel_ = 0.0
            if elapsed >= 5.0:
                self.active_command = None
            return True

        elif self.active_command == "SLOW_DOWN":
            self.vel_ = np.clip(self.vel_, -0.15, 0.2)
            if elapsed >= 2.0:
                self.active_command = None
            return False

        elif self.active_command == "PARK":
            self.vel_ = np.clip(self.vel_, -0.15, 0.2)
            if elapsed >= 2.0:
                self.active_command = None
            return False

        return False

    # -------------------------------------------------------------------------
    # Condensed QP assembly (Phi / Gamma / C batch formulation)
    # -------------------------------------------------------------------------
    def generate_phi_and_gamma(self, state, control_inputs):
        """Build the condensed-form prediction matrices Phi, Gamma, C.

        For each horizon step k, the model is linearized at the rolled-out
        nominal state/control pair, giving a per-step discrete pair
        (A_k, B_k) and an affine residual
        `g_k = x_next_nominal - A_k @ x_nominal - B_k @ u_nominal`
        (the part of the nonlinear step not captured by the local linear
        model). These are then propagated/accumulated so that the full
        predicted state sequence can be written as a single affine function
        of the decision variable U:

            X = Phi @ x0 + Gamma @ U + C

        where X and C stack 4-vectors per horizon step, and U stacks
        2-vectors (steering, acceleration) per horizon step.

        Args:
            state: current state [x, y, theta, v], used as the start of the
                nominal rollout (x0).
            control_inputs: shape (N_, 2) nominal control sequence used to
                linearize and roll out the trajectory.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                Phi (4*N_, 4), Gamma (4*N_, 2*N_), C (4*N_,).
        """
        A_list = []
        B_list = []
        g_list = []
        for i in range(self.N_):
            A, B = self.linearize_dynamics(state, control_inputs[i])
            x_nom = state.copy()
            u_nom = control_inputs[i].copy()

            x_next_nom = self.kinematic_model(x_nom, u_nom)

            g = (x_next_nom - A @ x_nom - B @ u_nom)

            g_list.append(g)
            A_list.append(A)
            B_list.append(B)
            state = x_next_nom

        Phi = np.zeros((4 * self.N_, 4))
        Gamma = np.zeros((4 * self.N_, 2 * self.N_))
        C = np.zeros(4 * self.N_)

        for k in range(self.N_):

            # Phi block for step k: product of A_0..A_k (state transition
            # from x0 to x_{k+1}).
            cummulative_mat = np.eye(4)
            for i in range(k + 1):
                cummulative_mat = cummulative_mat @ A_list[k - i]
            Phi[4 * k:4 * (k + 1), :] = cummulative_mat

            # Gamma block for step k, control j: how u_j influences x_{k+1},
            # propagated forward through A_{j+1}..A_k.
            for j in range(k + 1):

                temp = B_list[j]
                for i in range(j + 1, k + 1):
                    temp = A_list[i] @ temp
                Gamma[4 * k:4 * (k + 1), 2 * j:2 * (j + 1)] = temp

            # C block for step k: accumulated linearization residual g_j,
            # likewise propagated forward through A_{j+1}..A_k.
            c_k = np.zeros(4)

            for j in range(k + 1):

                temp = g_list[j]

                for i in range(j + 1, k + 1):
                    temp = A_list[i] @ temp

                c_k += temp

            C[4 * k:4 * (k + 1)] = c_k

        return Phi, Gamma, C

    def generate_control_constraints(self):
        """Box constraints on absolute steering angle and acceleration.

        Builds `G_u @ U <= h_u` encoding, for every horizon step,
        `steer_min <= steer_k <= steer_max` and `accel_min <= accel_k <= accel_max`.

        Returns:
            tuple[np.ndarray, np.ndarray]: (G_u, h_u).
        """
        I = np.eye(2 * self.N_)
        G_u = np.vstack((I, -I))
        U_max = np.array([[self.max_steering_angle_Right],
                           [self.max_acceleration_]] * self.N_)
        U_min = np.array([[self.max_steering_angle_Left],
                           [self.min_acceleration_]] * self.N_)
        h_u = np.vstack((U_max, -U_min))

        return G_u, h_u

    def generate_control_change_constraints(self, previous_control_inputs):
        """Rate (slew) constraints between consecutive control inputs.

        Limits `|u_k - u_{k-1}| <= [max_steering_change_, max_acceleration_change_]`
        for k = 1..N_-1, and additionally couples step 0 to the control
        input that was actually applied last cycle (`previous_control_inputs`),
        so the very first step of the new horizon can't jump
        discontinuously from what the actuator is currently doing.

        Args:
            previous_control_inputs: [steer, accel] applied on the previous
                cycle.

        Returns:
            tuple[np.ndarray, np.ndarray]: (G_d, h_d) stacking the upper
            and lower one-sided versions of the rate constraint.
        """
        G_d_1 = np.zeros((2 * self.N_, 2 * self.N_))
        h_d_1 = np.zeros((2 * self.N_, 1))
        for i in range(self.N_):
            if i == 0:
                G_d_1[2 * i:2 * (i + 1), 0:2] = np.eye(2)
                h_d_1[0:2, 0] = np.array([self.max_steering_change_, self.max_acceleration_change_]) + previous_control_inputs
            else:
                G_d_1[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = np.eye(2)
                G_d_1[2 * i:2 * (i + 1), 2 * (i - 1):2 * i] = -np.eye(2)
                h_d_1[2 * i:2 * (i + 1), 0] = np.array([self.max_steering_change_, self.max_acceleration_change_])

        G_d_2 = -G_d_1
        h_d_2 = np.zeros((2 * self.N_, 1))
        for i in range(self.N_):
            if i == 0:
                h_d_2[0:2, 0] = np.array([self.max_steering_change_, self.max_acceleration_change_]) - previous_control_inputs
            else:
                h_d_2[2 * i:2 * (i + 1), 0] = np.array([self.max_steering_change_, self.max_acceleration_change_])
        G_d = np.vstack((G_d_1, G_d_2))
        h_d = np.vstack((h_d_1, h_d_2))

        return G_d, h_d

    def generate_velocity_constraints(self, Phi, Gamma, C, state):
        """Linear constraints enforcing velocity bounds via the condensed model.

        Velocity is state element index 3 at every horizon step. `Cv`
        selects those entries out of the stacked state vector, so
        `Cv @ X = Cv @ (Phi @ state + Gamma @ U + C)` gives the predicted
        velocity at each step as an affine function of U -- this is
        rearranged into `G_v @ U <= h_v` for both the upper bound
        (`max_velocity_`) and lower bound (`min_velocity_`).

        Args:
            Phi, Gamma, C: condensed prediction matrices from
                `generate_phi_and_gamma`.
            state: current state, used as x0 in the condensed prediction.

        Returns:
            tuple[np.ndarray, np.ndarray]: (G_v, h_v).
        """
        Cv = np.zeros((self.N_, 4 * self.N_))

        for k in range(self.N_):
            Cv[k, 4 * k + 3] = 1.0

        G_v_pos = Cv @ Gamma
        h_v_pos = (self.max_velocity_ - Cv @ (Phi @ state + C))

        G_v_neg = -G_v_pos
        h_v_neg = -(self.min_velocity_ - Cv @ (Phi @ state + C))

        G_v = np.vstack([G_v_pos, G_v_neg])
        h_v = np.hstack([h_v_pos, h_v_neg])

        return G_v, h_v

    # -------------------------------------------------------------------------
    # Main control cycle
    # -------------------------------------------------------------------------
    def mpc_control(self, state, control_inputs, reference_trajectory):
        """Compute the control command for the current cycle.

        Execution order:
          1. Sign-reaction override (`act_on_sign`): if a STOP is active,
             short-circuits with a zero command immediately.
          2. Escape-maneuver continuation: if currently inside an escape
             dwell window, keep issuing the open-loop escape command rather
             than touching the QP at all.
          3. Escape-maneuver exit: once the dwell window has just elapsed,
             clear the nominal control sequence / warm start so the MPC
             restarts the QP from a clean slate.
          4. Build and solve the condensed QP for this cycle (see
             `generate_phi_and_gamma`, `generate_Q_and_R_matrices`, and the
             three constraint generators), after correcting the reference
             heading for 2*pi wraparound relative to the current predicted
             heading.
          5. Integrate the solved acceleration into `vel_`: during the
             startup phase this is an open-loop ramp (clipped, direction
             set by `reverse_start_`) gated on either reaching a velocity
             threshold or timing out after 3 s; once startup is complete,
             `vel_` is integrated directly from `u0[1]` every cycle.
          6. Predicted-collision check on the just-solved horizon
             (`predicted_path_safe`): SLOW clamps velocity, STOP aborts the
             MPC command in favour of an open-loop escape maneuver chosen
             by comparing forward/backward clearance probes.
          7. Goal-proximity velocity cap and goal-reached latch.

        Args:
            state: current state [x, y, theta, v].
            control_inputs: shape (N_, 2) nominal control sequence carried
                over from the previous cycle (used for linearization).
            reference_trajectory: shape (N_, 4) reference trajectory from
                `build_reference_horizon`.

        Returns:
            np.ndarray | None: [steering, acceleration] command for this
            cycle (acceleration is informational -- the actual commanded
            speed is `self.vel_`), or None if the QP failed to solve.
        """
        now = self.get_clock().now()

        if self.act_on_sign():
            return np.array([0.0, 0.0])

        if (self.escape_direction_ is not None
                and self.escape_until_time_ is not None
                and now < self.escape_until_time_):
            self.vel_ = (self.forward_speed_ if self.escape_direction_ > 0
                         else self.reverse_speed_)
            self.escape_steer_ = self._compute_escape_steering(
                self.state_[0], self.state_[1], self.state_[2], self.escape_direction_)
            return np.array([self.escape_steer_, 0.0])

        if self.escape_direction_ is not None and now >= self.escape_until_time_:
            self.control_input_ = np.zeros((self.N_, 2))
            self.prev_optimal_control_input_ = np.array([0.0, 0.0])
            self.prev_solution_ = None
            self._last_escape_steer_ = 0.0

        Phi, Gamma, C = self.generate_phi_and_gamma(state, control_inputs)

        Q, R = self.generate_Q_and_R_matrices()

        # Correct reference heading for 2*pi wraparound relative to the
        # nominal predicted heading, so the QP never "sees" an artificial
        # ~2*pi tracking error when the path heading wraps around.
        ref_flat = reference_trajectory.flatten()
        pred_flat = (Phi @ state + C)

        for k in range(self.N_):
            heading_diff = ref_flat[4 * k + 2] - pred_flat[4 * k + 2]
            ref_flat[4 * k + 2] = pred_flat[4 * k + 2] + np.arctan2(np.sin(heading_diff), np.cos(heading_diff))

        # Standard QP form: min 0.5*U'HU + f'U s.t. l <= G_ U <= u
        H = 2 * (Gamma.T @ Q @ Gamma + R)
        f = 2 * (Gamma.T @ Q @ (pred_flat - ref_flat))

        G_u, h_u = self.generate_control_constraints()
        G_d, h_d = self.generate_control_change_constraints(self.prev_optimal_control_input_)
        G_v, h_v = self.generate_velocity_constraints(Phi, Gamma, C, state)
        G_ = np.vstack((G_u, G_d, G_v))
        h_ = np.hstack((h_u.flatten(), h_d.flatten(), h_v.flatten()))

        H = 0.5 * (H + H.T)   # enforce exact symmetry for the QP solver
        f = np.asarray(f).flatten()
        l = -np.inf * np.ones(G_.shape[0])
        u = h_.flatten()

        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(H), q=f, A=sparse.csc_matrix(G_), l=l, u=u,
                   verbose=False, warm_starting=True, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000)

        if self.prev_solution_ is not None:
            prob.warm_start(x=self.prev_solution_)

        res = prob.solve()

        if res.info.status != "solved":
            self.get_logger().warn(f"MPC failed: {res.info.status}")
            return None

        # self.get_logger().info(f"OSQP status={res.info.status} " f"iter={res.info.iter} " f"obj={res.info.obj_val:.3f}")

        self.prev_solution_ = res.x.copy()
        U_opt = res.x

        U_opt_matrix = U_opt.reshape(self.N_, 2)

        # Roll out the nominal (open-loop) trajectory under the solved
        # control sequence, purely for visualization/debugging.
        x_nom = state.copy()
        nominal_states = []
        for k in range(self.N_):
            nominal_states.append(x_nom.copy())
            x_nom = self.kinematic_model(x_nom, U_opt_matrix[k])

        self.publish_nominal_path(np.array(nominal_states))

        u0 = U_opt[0:2]
        self.prev_optimal_control_input_ = u0.copy()
        U_opt_matrix = U_opt.reshape(self.N_, 2)
        # Shift the solved sequence forward by one step to seed next
        # cycle's linearization/warm-start nominal trajectory.
        self.control_input_[:-1] = U_opt_matrix[1:]
        self.control_input_[-1] = U_opt_matrix[-1]

        if not self.startup_complete_:
            # Startup ramp: rather than trusting the integrated commanded
            # velocity from a standstill, ramp vel_ open-loop using the
            # magnitude of the solved acceleration until real motion is
            # detected (or a timeout elapses, in case the robot is stuck).
            if self.startup_time_ is None:
                self.startup_time_ = self.get_clock().now().nanoseconds * 1e-9

            if self.reverse_start_:
                self.vel_ -= abs(u0[1]) * self.dt_
            else:
                self.vel_ += abs(u0[1]) * self.dt_

            self.vel_ = np.clip(self.vel_, self.min_velocity_, self.max_velocity_)

            elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.startup_time_
            if abs(self.state_[3]) > self.STARTUP_VEL_THRESHOLD or elapsed > 3.0:
                if elapsed > 3.0:
                    self.get_logger().warn(
                        f"Startup timeout — vel={self.state_[3]:.3f} m/s, robot may be stuck"
                    )
                self.startup_complete_ = True
                self.startup_time_ = None
                self.get_logger().info(f"Startup complete at {self.state_[3]:.3f} m/s")
        else:
            self.vel_ = self.vel_ + u0[1] * self.dt_
            self.vel_ = float(np.clip(self.vel_, self.min_velocity_, self.max_velocity_))

        X_pred = Phi @ state + Gamma @ U_opt + C
        status, index = self.predicted_path_safe(X_pred)
        self.publish_predicted_path(X_pred.reshape(self.N_, 4))

        if status == "SLOW":
            self.vel_ = np.clip(self.vel_, -0.25, 0.3)

        elif status == "STOP":

            self.get_logger().warn(f"Predicted collision at step {index}")

            x = self.state_[0]
            y = self.state_[1]
            yaw = self.state_[2]

            # Probe clearance just ahead of and just behind the vehicle to
            # decide which direction has more room to escape into.
            front_x = x + self.probe_distance_ * np.cos(yaw)
            front_y = y + self.probe_distance_ * np.sin(yaw)
            back_x = x - self.probe_distance_ * np.cos(yaw)
            back_y = y - self.probe_distance_ * np.sin(yaw)

            front = self.clearance_at(front_x, front_y)
            back = self.clearance_at(back_x, back_y)

            if front < back:
                chosen_dir = -1
            elif back < front:
                chosen_dir = +1
            else:
                chosen_dir = -1   # tie -> reverse bias

            self.escape_direction_ = chosen_dir
            self.escape_until_time_ = now + rclpy.duration.Duration(seconds=self.ESCAPE_DWELL_SEC_)
            self.vel_ = self.forward_speed_ if chosen_dir > 0 else self.reverse_speed_
            self.escape_steer_ = self._compute_escape_steering(x, y, yaw, chosen_dir)

            return np.array([self.escape_steer_, 0.0])

        self.escape_direction_ = None
        self.escape_until_time_ = None

        distance_to_goal = np.linalg.norm(self.state_[0:2] - np.array([self.path_x_[-1], self.path_y_[-1]]))

        if distance_to_goal < self.SLOWDOWN_RADIUS_:
            t = distance_to_goal / self.SLOWDOWN_RADIUS_
            vel_cap = self.GOAL_VEL_ + t * (0.52 - self.GOAL_VEL_)
            self.vel_ = float(np.clip(self.vel_, -vel_cap, vel_cap))

        if distance_to_goal < 0.05:
            if not self.goal_reached_:
                self.get_logger().info("Goal reached!")
            self.goal_reached_ = True
        else:
            self.goal_reached_ = False

        return u0

    # -------------------------------------------------------------------------
    # Visualization / debug publishers
    # -------------------------------------------------------------------------
    def update_path_trace(self):
        """Append the current pose to the executed-path trace and publish it."""
        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.state_[0]
        pose.pose.position.y = self.state_[1]
        pose.pose.position.z = 0.0
        self.path_trace_.header.stamp = pose.header.stamp
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0

        self.path_trace_.poses.append(pose)
        self.path_trace_pub_.publish(self.path_trace_)

    def publish_predicted_path(self, X_pred):
        """Publish the closed-loop QP-predicted horizon as a nav_msgs/Path.

        Args:
            X_pred: shape (N_, 4) predicted state sequence.
        """
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for k in range(X_pred.shape[0]):

            pose = PoseStamped()

            pose.header = path_msg.header

            pose.pose.position.x = float(X_pred[k, 0])
            pose.pose.position.y = float(X_pred[k, 1])
            pose.pose.position.z = 0.0

            path_msg.poses.append(pose)

        self.pred_path_pub_.publish(path_msg)

    def publish_reference_path(self, reference_trajectory):
        """Publish the reference horizon as a nav_msgs/Path.

        Args:
            reference_trajectory: shape (N_, 4) reference trajectory.
        """
        path_msg = Path()

        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for k in range(reference_trajectory.shape[0]):

            pose = PoseStamped()

            pose.header = path_msg.header

            pose.pose.position.x = float(reference_trajectory[k, 0])
            pose.pose.position.y = float(reference_trajectory[k, 1])
            pose.pose.position.z = 0.0

            path_msg.poses.append(pose)

        self.reference_path_pub_.publish(path_msg)

    def publish_nominal_path(self, nominal_states):
        """Publish the open-loop nominal rollout (under the solved U) as a nav_msgs/Path.

        Args:
            nominal_states: iterable of [x, y, theta, v] states.
        """
        path_msg = Path()

        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for state in nominal_states:

            pose = PoseStamped()

            pose.header = path_msg.header

            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            pose.pose.position.z = 0.0

            path_msg.poses.append(pose)

        self.nominal_path_pub_.publish(path_msg)

    # -------------------------------------------------------------------------
    # Odometry callback (main entry point, runs at the odometry rate)
    # -------------------------------------------------------------------------
    def odom_callback(self, msg):
        """Update the state estimate and drive the controller for one cycle.

        Order of operations:
          1. Update `state_` from the filtered odometry message (position,
             yaw extracted from quaternion, forward velocity) and append to
             the executed-path trace.
          2. Position-jump handling: if a jump is detected and we are not
             already waiting on a replan, stop the vehicle and request a
             Hybrid A* replan, then return early. While waiting on a
             pending replan, keep publishing a zero-velocity Twist and
             return early every cycle.
          3. Otherwise, if a path is loaded, build the reference horizon,
             run `mpc_control`, and publish the resulting command on both
             `/velocity_steer` (Float64MultiArray: [vel, steer]) and
             `/test_twist` (geometry_msgs/Twist) -- or log a warning if the
             QP failed to solve.

        Args:
            msg: nav_msgs/Odometry, expected on `/odometry/filtered`.
        """
        self.state_[0] = msg.pose.pose.position.x
        self.state_[1] = msg.pose.pose.position.y
        _, _, yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
                                            msg.pose.pose.orientation.y,
                                            msg.pose.pose.orientation.z,
                                            msg.pose.pose.orientation.w])
        self.state_[2] = yaw
        self.state_[3] = msg.twist.twist.linear.x
        self.update_path_trace()

        if self.check_position_jump() and not self.waiting_replan_:
            self.waiting_replan_ = True
            self.startup_complete_ = False
            self.vel_ = 0.0
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.twist_pub_.publish(stop_msg)
            self.request_replan()
            return

        if self.waiting_replan_:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.twist_pub_.publish(stop_msg)
            return

        if len(self.path_x_) > 0:
            reference_trajectory = self.build_reference_horizon()
            self.publish_reference_path(reference_trajectory)
            optimal_control_input = self.mpc_control(self.state_, self.control_input_, reference_trajectory)
            if optimal_control_input is not None:
                control_msg = Float64MultiArray()
                control_msg.data = [self.vel_, optimal_control_input[0]]
                self.control_pub_.publish(control_msg)
                msg = Twist()
                msg.linear.x = self.vel_
                msg.angular.z = optimal_control_input[0]
                self.twist_pub_.publish(msg)
            else:
                self.get_logger().warn("MPC failed to find a solution.")


def main(args=None):
    """Standard ROS 2 node entry point: init, spin MPCNode, shutdown."""
    rclpy.init(args=args)
    node = MPCNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
