#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from smbus2 import SMBus


class AS5600RPMNode(Node):

    def __init__(self):
        super().__init__('as5600_rpm_node')

        # Publisher
        self.publisher_ = self.create_publisher(
            Float64,
            '/encoder/rpm',
            10
        )

        # I2C setup (Jetson confirmed bus 7)
        self.bus = SMBus(7)
        self.address = 0x36

        self.MAX_TICKS = 4096  # 12-bit encoder

        self.prev_angle = None
        self.prev_time = self.get_clock().now()

        # filter state (EMA smoothing)
        self.rpm_filtered = 0.0
        self.alpha = 0.2

        self.timer = self.create_timer(
            1.0 / 200.0,
            self.timer_callback
        )

        self.get_logger().info("AS5600 RPM Node started (bus 7)")

    # -----------------------------
    # Read angle (correct registers)
    # -----------------------------
    def read_raw_angle(self):
        high = self.bus.read_byte_data(self.address, 0x0E)
        low  = self.bus.read_byte_data(self.address, 0x0F)

        return ((high << 8) | low) & 0x0FFF

    # -----------------------------
    # Magnet status check
    # -----------------------------
    def magnet_ok(self):
        status = self.bus.read_byte_data(self.address, 0x0B)

        md = (status >> 5) & 1  # magnet detected
        ml = (status >> 4) & 1  # too weak
        mh = (status >> 3) & 1  # too strong

        return md == 1 and ml == 0 and mh == 0

    # -----------------------------
    # Main loop
    # -----------------------------
    def timer_callback(self):

        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds / 1e9

        if dt <= 0:
            return

        # MAGNET VALIDATION (fix garbage data)
        try:
            if not self.magnet_ok():
                msg = Float64()
                msg.data = 0.0
                self.publisher_.publish(msg)

                self.prev_angle = None
                self.prev_time = now
                return

            current_angle = self.read_raw_angle()

        except Exception as e:
            self.get_logger().error(f"I2C error: {e}")
            return

        # init
        if self.prev_angle is None:
            self.prev_angle = current_angle
            self.prev_time = now
            return

        delta = current_angle - self.prev_angle

        # unwrap rollover
        if delta > self.MAX_TICKS / 2:
            delta -= self.MAX_TICKS
        elif delta < -self.MAX_TICKS / 2:
            delta += self.MAX_TICKS

        # -----------------------------
        # DEAD BAND (removes jitter)
        # -----------------------------
        if abs(delta) < 5:
            rpm = 0.0
        else:
            rotations = delta / self.MAX_TICKS
            rpm = (rotations / dt) * 60.0

        # -----------------------------
        # EMA smoothing filter
        # -----------------------------
        self.rpm_filtered = (
            self.alpha * rpm +
            (1 - self.alpha) * self.rpm_filtered
        )

        # publish
        msg = Float64()
        msg.data = float(self.rpm_filtered)
        self.publisher_.publish(msg)

        # update state
        self.prev_angle = current_angle
        self.prev_time = now

    def destroy_node(self):
        self.bus.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = AS5600RPMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
