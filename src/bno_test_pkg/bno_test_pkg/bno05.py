import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Imu

from smbus2 import SMBus, i2c_msg
import struct
import time

class BNO055I2CNode(Node):

    def __init__(self):
        super().__init__('bno055_i2c_node')

        # Core IMU Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)

        # Diagnostic/Debug Publishers
        self.yaw_pub = self.create_publisher(Float64, 'bno/yaw', 10)
        self.gyro_pub = self.create_publisher(Vector3, 'bno/gyro', 10)
        self.accel_pub = self.create_publisher(Vector3, 'bno/accel', 10)
        
        # Jetson I2C Settings (Typically bus 1 or 8 on Jetson Nano/Xavier/Orin)
        # Default BNO055 I2C address is 0x28 (can be 0x29 if COM pin is pulled high)
        self.i2c_bus_number = 1 
        self.bno_address = 0x28

        # Initialize I2C Bus Connection
        try:
            self.bus = SMBus(self.i2c_bus_number)
            self.init_hardware()
        except Exception as e:
            self.get_logger().error(f"Failed to open I2C bus {self.i2c_bus_number}: {e}")
            return

        # 50 Hz Timer Loop (0.02 seconds)
        self.timer = self.create_timer(0.03, self.timer_callback)

    def write_reg(self, reg, value):
        """Writes a byte to a specific BNO055 register over I2C."""
        try:
            self.bus.write_byte_data(self.bno_address, reg, value)
        except Exception as e:
            self.get_logger().error(f"I2C Write failed to reg {hex(reg)}: {e}")

    def read_regs(self, reg, length):
        """Reads a continuous block of registers via I2C combined transactions."""
        try:
            write = i2c_msg.write(self.bno_address, [reg])
            read = i2c_msg.read(self.bno_address, length)
            self.bus.i2c_rdwr(write, read)
            return bytes(read)
        except Exception as e:
            self.get_logger().warn(f"I2C Read failed at reg {hex(reg)}: {e}")
            return b''

    def init_hardware(self):
        self.get_logger().info("Initializing BNO055 over I2C...")
        
        # 1. Switch to CONFIG Mode (0x00) to clear running tasks
        self.write_reg(0x3D, 0x00)
        time.sleep(0.05)
        
        # 2. Trigger System Reset
        self.write_reg(0x3F, 0x20)
        time.sleep(0.7) # Wait for chip to reboot completely
        
        # 3. Ensure Normal Power Mode
        self.write_reg(0x3E, 0x00)
        time.sleep(0.05)
        
        # 4. Set Units Configuration (0x3B) -> Android, Celsius, Rad/s, m/s^2
        self.write_reg(0x3B, 0x02)
        time.sleep(0.05)

        # 5. Set operational fusion mode to NDOF (0x08)
        self.get_logger().info("Setting BNO055 to Auto-Filtered NDOF Mode (0x08)...")
        self.write_reg(0x3D, 0x08)
        time.sleep(0.2) # Allow fusion engine to initialize

        # Verification Loop
        verified = False
        for _ in range(10):
            mode = self.read_regs(0x3D, 1)
            if mode and (mode[0] & 0x0F) == 0x08:
                self.get_logger().info("BNO055 successfully confirmed in NDOF mode via I2C.")
                verified = True
                break
            time.sleep(0.05)
            
        if not verified:
            self.get_logger().error("BNO055 rejected NDOF mode assignment. Running in fallback.")

    def timer_callback(self):
        # Read everything from Gyro (0x14) up to Linear Accel (0x2D) -> 26 continuous bytes
        raw_data = self.read_regs(0x14, 26)
        
        if len(raw_data) < 26:
            return # Drop bad/incomplete frames

        current_time = self.get_clock().now()

        # --- PARSE DATA BLOCK ---
        # Gyro (6 bytes @ offset 0). Scale factor: 16 LSB = 1 rad/s
        gyro_raw = struct.unpack('<hhh', raw_data[0:6])
        gyro_old = [(gyro_raw[0] / 16.0) * -1.0, (gyro_raw[1] / 16.0) * -1.0, (gyro_raw[2] / 16.0) * -1.0]
        
        # Rotate Gyro 90 deg CCW around Z: X_new = -Y_old, Y_new = X_old
        gyro = [-gyro_old[1], gyro_old[0], gyro_old[2]]

        # Euler Heading (2 bytes @ offset 6). Scale factor: 16 LSB = 1 degree
        heading_raw = struct.unpack('<h', raw_data[6:8])[0]
        heading = (heading_raw / 16.0) + 90.0  # Add 90 degrees for CCW rotation
        # Normalize heading to standard +/- 180 coordinate orientation
        if heading > 180.0:
            heading -= 360.0

        # Quaternions (8 bytes @ offset 12). Scale factor: 16384 LSB = 1 unit
        q_raw = struct.unpack('<hhhh', raw_data[12:20])
        scale_q = 16384.0
        qw_old, qx_old, qy_old, qz_old = q_raw[0] / scale_q, q_raw[1] / scale_q, q_raw[2] / scale_q, q_raw[3] / scale_q

        # Apply 90 degree CCW rotation around Z-axis to the Quaternion
        # (Derived from multiplying the raw quaternion by: w=0.7071, x=0, y=0, z=0.7071)
        qw = 0.7071 * qw_old - 0.7071 * qz_old
        qx = 0.7071 * qx_old - 0.7071 * qy_old
        qy = 0.7071 * qx_old + 0.7071 * qy_old
        qz = 0.7071 * qw_old + 0.7071 * qz_old

        # Linear Acceleration (6 bytes @ offset 20). Scale factor: 100 LSB = 1 m/s^2
        accel_raw = struct.unpack('<hhh', raw_data[20:26])
        lin_accel_old = [accel_raw[0] / 100.0, accel_raw[1] / 100.0, accel_raw[2] / 100.0]
        
        # Rotate Accel 90 deg CCW around Z: X_new = -Y_old, Y_new = X_old
        lin_accel = [-lin_accel_old[1], lin_accel_old[0], lin_accel_old[2]]

        # --- CONSTRUCT AND PUBLISH IMU DATA ---
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = "imu_link"
        
        # Orientation (Quaternions)
        imu_msg.orientation.x = qx
        imu_msg.orientation.y = qy
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        
        # Angular Velocity
        imu_msg.angular_velocity.x = gyro[0]
        imu_msg.angular_velocity.y = gyro[1]
        imu_msg.angular_velocity.z = gyro[2]
        
        # Linear Acceleration
        imu_msg.linear_acceleration.x = lin_accel[0]
        imu_msg.linear_acceleration.y = lin_accel[1]
        imu_msg.linear_acceleration.z = lin_accel[2]

        # Statistical Covariance Overlays
        imu_msg.orientation_covariance = [0.002, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.002]
        imu_msg.angular_velocity_covariance = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
        imu_msg.linear_acceleration_covariance = [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04]

        # Publish Everything
        self.imu_pub.publish(imu_msg)
        self.yaw_pub.publish(Float64(data=heading))
        self.gyro_pub.publish(Vector3(x=gyro[0], y=gyro[1], z=gyro[2]))
        self.accel_pub.publish(Vector3(x=lin_accel[0], y=lin_accel[1], z=lin_accel[2]))


def main(args=None):
    rclpy.init(args=args)
    node = BNO055I2CNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
