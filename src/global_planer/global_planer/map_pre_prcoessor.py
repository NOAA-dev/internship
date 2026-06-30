#!/usr/bin/env python3
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from PIL import Image
 
 
class MapResizer(Node):
    def __init__(self):
        super().__init__("map_resizer") 

        self.declare_parameter("map_path", "/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/track_3.png")
        self.declare_parameter("planning_size_x", 600)
        self.declare_parameter("planning_size_y", 500)
        self.declare_parameter("inflation_kernel_size", 5)
        self.map_path = self.get_parameter("map_path").value
        self.planning_size_x = self.get_parameter("planning_size_x").value
        self.planning_size_y = self.get_parameter("planning_size_y").value
        self.inflation_kernel_size = self.get_parameter("inflation_kernel_size").value

    def proccess_image(self):

        # Load image
        img = cv2.imread(self.map_path)
        
        # Resize for planning
        img = cv2.resize(img,(self.planning_size_x, self.planning_size_y),interpolation=cv2.INTER_AREA)
        img = cv2.flip(img,1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Detect ONLY near-black walls
        # black_mask = cv2.inRange(img,(0, 0, 0),(150, 150, 150))
        # black_mask_mpc = cv2.inRange(img,(0, 0, 0),(150, 150, 150))
        black_mask = img
        black_mask_mpc = img
        # Convert to occupancy grid
        occupancy_grid = np.ones_like(black_mask, dtype=np.uint8)
        occupancy_grid[black_mask > 0] = 0

        occupancy_grid_mpc = np.ones_like(black_mask_mpc, dtype=np.uint8)
        occupancy_grid_mpc[black_mask_mpc > 0] = 0

        # Inflate obstacles
        kernel = np.ones((self.inflation_kernel_size, self.inflation_kernel_size), np.uint8)
        occupancy_grid = cv2.dilate(occupancy_grid,kernel,iterations=1)

        # Save
        np.save("/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/occupancy_grid.npy", occupancy_grid)
        np.save("/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/occupancy_grid_mpc.npy", occupancy_grid_mpc)

        debug_img = occupancy_grid * 255

        cv2.imwrite(
            "/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/debug_occupancy.png",
            debug_img
        )



        # Load your NumPy map array (where 1 = wall, 0 = free)
        map_array = np.load("/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/occupancy_grid_mpc.npy")
        
        # Create a viewable PGM array formatted for ROS Nav2
        # Initialize everything as White (255 - Free Space)
        ros_map_data = np.full_like(map_array, 255, dtype=np.uint8)
        
        # Set walls to Black (0 - Occupied Space)
        ros_map_data[map_array == 1] = 0

        # Save as PGM
        # Flip vertically for ROS coordinate system
        ros_map_data = np.flipud(ros_map_data)
        image = Image.fromarray(ros_map_data, mode='L')
        image.save("/home/car-02/working_v1/src/robot_maps/maps/iitb_maps/map.pgm")
        self.get_logger().info("PGM Map successfully saved for Nav2!")

        self.get_logger().info("size of map is : " + str(ros_map_data.shape))



 
def main(args=None):
    rclpy.init(args=args)
    node = MapResizer() 
    node.proccess_image()
    rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
