#!/usr/bin/env python3
"""
ROS2 node that logs Odometry, PoseStamped, and Path messages to CSV files.

Subscribes to:
  /odometry/filtered   (nav_msgs/Odometry)        - always publishing
  /apriltag/tag2/pose  (geometry_msgs/PoseStamped) - intermittent (IPS visibility)
  /path_vis            (nav_msgs/Path)             - 10 Hz, even if unchanged

Subscriptions in ROS2 don't require the publisher to exist at startup -
each callback just fires whenever that topic's publisher is up and
publishing. So /apriltag/tag2/pose rows simply stop appearing when the
tag drops out of view, and resume when it's back - no extra logic needed
for "when available".

Each topic gets its own CSV (different rates/availability -> trying to
merge them into one row per timestamp would mean lots of empty/duplicated
fields). Files are flushed after every write so you can tail -f them or
kill the node anytime without losing data.
"""

import os
import csv
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

# NOTE: assumed "/home/chirag/csv_files" (no space) - change here if you
# actually meant a folder literally named "csv files".
OUTPUT_DIR = "/home/car-02/working_v1/csv_files"


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class CsvLogger(Node):

    def __init__(self):
        super().__init__('csv_logger_node')

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # BEST_EFFORT subscriber is compatible with both RELIABLE and
        # BEST_EFFORT publishers, so this connects regardless of how the
        # publishers were set up. If you need guaranteed delivery and know
        # the publishers are RELIABLE, switch this to RELIABLE.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- /odometry/filtered ---
        self.odom_file = open(
            os.path.join(OUTPUT_DIR, 'odometry_filtered.csv'), 'w', newline='')
        self.odom_writer = csv.writer(self.odom_file)
        self.odom_writer.writerow([
            'wall_time', 'stamp_sec',
            'pos_x', 'pos_y', 'pos_z',
            'quat_x', 'quat_y', 'quat_z', 'quat_w',
            'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
            'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
        ])
        self.odom_file.flush()
        self.create_subscription(Odometry, '/odometry/filtered', self.odom_cb, qos)

        # --- /apriltag/tag2/pose ---
        self.pose_file = open(
            os.path.join(OUTPUT_DIR, 'apriltag_tag2_pose.csv'), 'w', newline='')
        self.pose_writer = csv.writer(self.pose_file)
        self.pose_writer.writerow([
            'wall_time', 'stamp_sec', 'frame_id',
            'pos_x', 'pos_y', 'pos_z',
            'quat_x', 'quat_y', 'quat_z', 'quat_w',
        ])
        self.pose_file.flush()
        self.create_subscription(PoseStamped, '/apriltag/tag2/pose', self.pose_cb, qos)

        # --- /path_vis ---
        self.path_file = open(
            os.path.join(OUTPUT_DIR, 'path_vis.csv'), 'w', newline='')
        self.path_writer = csv.writer(self.path_file)
        self.path_writer.writerow([
            'wall_time', 'stamp_sec', 'frame_id',
            'num_poses', 'poses_json',
        ])
        self.path_file.flush()
        self.create_subscription(Path, '/path_vis', self.path_cb, qos)

        self.get_logger().info(f'Logging CSVs to {OUTPUT_DIR}')

    # ------------------------------------------------------------------

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        av = msg.twist.twist.angular
        self.odom_writer.writerow([
            self.get_clock().now().nanoseconds * 1e-9,
            stamp_to_sec(msg.header.stamp),
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z,
        ])
        self.odom_file.flush()

    def pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.pose_writer.writerow([
            self.get_clock().now().nanoseconds * 1e-9,
            stamp_to_sec(msg.header.stamp),
            msg.header.frame_id,
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w,
        ])
        self.pose_file.flush()

    def path_cb(self, msg: Path):
        poses = [
            {'x': ps.pose.position.x, 'y': ps.pose.position.y, 'z': ps.pose.position.z}
            for ps in msg.poses
        ]
        self.path_writer.writerow([
            self.get_clock().now().nanoseconds * 1e-9,
            stamp_to_sec(msg.header.stamp),
            msg.header.frame_id,
            len(msg.poses),
            json.dumps(poses),
        ])
        self.path_file.flush()

    def destroy_node(self):
        for f in (self.odom_file, self.pose_file, self.path_file):
            try:
                f.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CsvLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()