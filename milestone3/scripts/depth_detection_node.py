#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import time
import os

class DepthDetectionNode(Node):
    def __init__(self):
        super().__init__('depth_detection_node')

        # get camera images from the simulator
        self.image_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.image_callback, 10)
        self.image_sub  # prevent unused variable warning

        # get lidar scan for emergency stop
        #self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        #self.scan_subscription  # prevent unused variable warning

        # Publisher for drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.bridge = CvBridge()

        # declare our parameters HERE:
        self.declare_parameter('num_laps', 5) # TODO: if num_laps==-1 then run indefinitely
        self.num_laps = self.get_parameter('num_laps').get_parameter_value().integer_value
        # just to check if the parameter is being read correctly from the launch file:
        self.get_logger().info(f'num_laps = {self.num_laps}')
        self.lap_time_seconds = 25 # TODO: if lap_time_seconds==-1 then run indefinitely

        # Display Image:
        self.display_image = False
        self.output_folder = "/home/jetson/sim_ws/src/team-a3-redbull/milestone3/images" # TODO

        # Emergency stop flag and speed
        self.stop = False
        self.stop_threshold = 0.1

        # Max speed of the car
        self.max_throttle = 1.5

        # Max steering angle
        self.max_steering = np.deg2rad(90)

        # PID controller parameters
        self.pid_kp = 0.003
        self.pid_kd = 0.0
        self.pid_ki = 0
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.last_time = time.time()

    def process_depth_image(self, depth_image):
        # Get image dimensions
        height, width = depth_image.shape
        
        # Define detection box
        box_width = int(width * 0.6)
        box_height = int(height * 0.05)
        center_x = width // 2 
        center_y = height // 2 + 20
        box_x1 = max(center_x - box_width // 2, 0)
        box_y1 = max(center_y - box_height // 2, 0)
        box_x2 = min(box_x1 + box_width, width)
        box_y2 = min(box_y1 + box_height, height)
        
        # Extract detection box region
        detection_region = depth_image[box_y1:box_y2, box_x1:box_x2]
        
        # Find the pixel with maximum depth (furthest away)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(detection_region)
        furthest_x = box_x1 + max_loc[0]
        furthest_y = box_y1 + max_loc[1]
        
        # Compute PID steering control based on horizontal error:
        # Use the difference between the furthest pixel's x coordinate and the box center.
        detection_center_x = (box_x1 + box_x2) // 2
        error = furthest_x - detection_center_x

        current_time = time.time()
        dt = current_time - self.last_time if current_time - self.last_time > 0 else 1e-6
        self.pid_integral += error * dt
        derivative = (error - self.pid_prev_error) / dt

        steering = self.pid_kp * error + self.pid_ki * self.pid_integral + self.pid_kd * derivative

        self.pid_prev_error = error
        self.last_time = current_time

        steering = np.clip(steering, -self.max_steering, self.max_steering)

        steering = np.square(steering) * np.sign(steering)  # Square the steering value

        # reduce throttle (speed) when the error is large since its usually a big turn
        throttle = self.max_throttle * (1 - min(abs(error) / detection_center_x, 1))

        # throttle = self.get_speed(steering)

        # Prepare visualization
        if self.display_image:
            # Normalize depth image for display
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Draw the detection box
            cv2.rectangle(depth_colormap, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
            # Draw a circle at the furthest point
            cv2.circle(depth_colormap, (furthest_x, furthest_y), 5, (0, 0, 255), -1)
            # Draw an arrow from the detection box center to the furthest point
            cv2.arrowedLine(depth_colormap, (detection_center_x, (box_y1+box_y2)//2),
                            (furthest_x, furthest_y), (255, 255, 255), 2)
            # Overlay the computed steering and throttle values
            cv2.putText(depth_colormap, f"Steering: {np.rad2deg(steering):.2f}  Throttle: {throttle:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Save the processed image instead of displaying
            os.makedirs(self.output_folder, exist_ok=True)  # Ensure the folder exists
            # filename = os.path.join(self.output_folder, f"processed_{int(time.time()*1000)}.png") # Save with unique timestamp
            filename = os.path.join(self.output_folder, "processed_image.png")  # Fixed filename
            cv2.imwrite(filename, depth_colormap)

            # # Display the processed image
            # cv2.imshow('Depth', depth_colormap)

        return -steering, throttle
    
    # def get_speed(self, angle):
    #     """
    #     Sets the speed of the car based on the steering angle.
    #     """
    #     if 0 <= abs(np.rad2deg(angle)) <= 10:
    #         speed = 1.5
    #     elif 10 < abs(np.rad2deg(angle)) <= 20:
    #         speed = 1.0
    #     elif 20 < abs(np.rad2deg(angle)) <= 30:
    #         speed = 0.7
    #     else:
    #         speed = 0.5
    #     return speed

    def publish_control(self, steering, throttle):
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = steering
            drive_msg.drive.speed = throttle
            self.drive_pub.publish(drive_msg)
            self.speed = throttle

    def image_callback(self, msg):
        if self.stop:
            return
        
        if self.check_laps():
            return
        
        # Convert ROS Image to OpenCV format
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)

        # Process the depth image
        steering, throttle = self.process_depth_image(depth_image)

        # Publish the computed steering and throttle values
        self.publish_control(steering, throttle)

        cv2.waitKey(1)
        
        self.get_logger().info(f"Computed steering: {np.rad2deg(steering):.3f}. Computed throttle: {throttle:.3f}")

    import time

    def check_laps(self) -> bool:
        """
        Stops the car if the number of laps has been reached.
        Returns True if the car should stop, False otherwise.
        """
        # If num_laps is -1 or lap_time_second is -1, the car should run indefinitely
        if self.num_laps == -1 or self.lap_time_seconds == -1:
            return False

        # Initialize start_time if it's the first time calling this function
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
            self.elapsed_laps = 0  # Track number of completed laps

        # Calculate elapsed time since the start
        elapsed_time = time.time() - self.start_time

        # Check if another lap should be counted
        if elapsed_time >= (self.elapsed_laps + 1) * self.lap_time_seconds:
            self.elapsed_laps += 1
            print(f"Lap {self.elapsed_laps}/{self.num_laps} completed.")

        # If the required number of laps is reached, stop the car
        if self.elapsed_laps >= self.num_laps:
            self.publish_control(0.0, 0.0)
            print("Stopping car. Lap limit reached.")
            return True
        
        return False
    
    def safety_stop(self, scan_msg) -> bool:
            """
            Implements emergency stop if the car is too close to an obstacle.
            """
            ranges = scan_msg.ranges
            print(ranges)
            for i in range(540-5, 540+5):
                if ranges[i] < self.stop_threshold:
                    self.publish_control(0.0, 0.0)
                    self.get_logger().info("EMERGENCY STOP")
                    return True
            return False
    
    def scan_callback(self, scan_msg):
        """
        Callback function for LaserScan messages.
        Implements gap follow and publishes the control message.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        print(scan_msg.ranges)
        if self.safety_stop(scan_msg):
            self.stop = True
        else:
            self.stop = False

        return

def main(args=None):
    rclpy.init(args=args)
    node = DepthDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()