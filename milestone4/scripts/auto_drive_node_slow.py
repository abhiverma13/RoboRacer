#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

import numpy as np
from sensor_msgs.msg import LaserScan, Image
from ackermann_msgs.msg import AckermannDriveStamped

import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import os
import csv

class AutoDrive(Node):
    """ 
    Implement auto driving on the car.
    """
    def __init__(self):
        super().__init__('auto_drive_node')

        lidarscan_topic = '/scan'
        camera_topic = '/camera/camera/color/image_raw'
        drive_topic = '/drive'

        # Create subscribers and publishers
        self.publisher_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.image_subscription = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.image_subscription # prevent unused variable warning
        self.bridge = CvBridge()
        self.scan_subscription = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.scan_subscription  # prevent unused variable warning

        self.lidar_csv_path = "/home/jetson/f1tenth_ws/src/team-a3-redbull/milestone4/lidar_data.csv"
        self.csv_header_written = False

        ## DRIVING PARAMETERS ##
        # Safety stop
        self.stop_threshold = 2.0
        self.stop = False
        self.largest_gap_distance_history = []
        self.largest_gap_distance_history_size = 20
        self.largest_gap_history = []
        self.largest_gap_history_size = 5
        
        # Configuration Parameters
        self.speed = 0.0
        self.speed_history = []
        self.speed_history_size = 10
        self.min_gap_distance = 0.5
        self.bubble_radius = 0.45
        self.range_angle = np.deg2rad(110)
        self.square_threshold = np.deg2rad(7)
        self.angle_factor = 1.0
        self.large_angle_threshold = np.deg2rad(10)
        self.large_angle_factor = 1.2

        # Variables for truncated range
        self.isTruncated = False
        self.truncated_start_index = 0
        self.truncated_end_index = 0

        # PID Controller Parameters
        self.kp = 0.65
        self.kd = 0.15
        self.ki = 0.0

        self.pid_integral = 0.0
        self.pid_prev_error = 0.0

        # Fixed control frequency
        self.dt = 0.05  # 20 Hz

        # # Variables for smoothing steering
        # self.steering_history = []
        # self.steering_history_size = 10

        ## IMAGE PARAMETERS ##
        # Display Image
        self.display_image = False
        self.output_folder = "/home/jetson/f1tenth_ws/src/team-a3-redbull/milestone4/reference_image"

        self.declare_parameter('num_laps', 3) # if num_laps==-1 then run indefinitely
        self.num_laps = self.get_parameter('num_laps').get_parameter_value().integer_value

        # Lap counting / reference image variables
        self.lap_count = 0
        self.ignoring = False
        self.ignoring_start_time = 0.0
        self.ignoring_duration = 5.0  # seconds to ignore after counting a lap
        self.detect_threshold = 0.8  # threshold for template matching

        self.reference_set = False   # Flag to indicate if the reference image has been stored
        self.processed_ref = None    # Will hold the processed reference image

        # Debugging
        self.print_speeds = True
        self.print_angles = True
        self.print_safety_stop_thresholds = True
        self.print_lap_timer = True

    def get_speed(self, angle, best_point_index, ranges):
        """
        Computes the desired speed of the car based on the steering angle and Lidar range data.
        
        Preconditions:
          - angle is in radians.
          - ranges is a list of distances (float values) corresponding to the preprocessed LiDAR scan.
        
        Postconditions:
          - Returns a speed value capped at 3.0.
        
        Rep invariant:
          - The computed speed respects the upper bound and matches one of the preset speeds based on the angle range.
        """
        if 0 <= abs(np.rad2deg(angle)) <= 2:
            speed = 1.6 + math.exp(0.04 * abs(ranges[best_point_index]))
            if abs(ranges[best_point_index]) < 3:
                speed -= 1.0
            if self.print_speeds:
                print("speed 1: ", speed)
        elif 2 < abs(np.rad2deg(angle)) <= 10:
            speed = 2.0
            if abs(ranges[best_point_index]) < 3:
                speed -= 1.0
            if self.print_speeds:
                print("speed 2: ", speed)
        elif 10 < abs(np.rad2deg(angle)) <= 15:
            speed = 1.8
            if abs(ranges[best_point_index]) < 2:
                speed -= 1.0
            if self.print_speeds:       
              print("speed 3: ", speed)
        else:
            speed = 1.5
            if abs(ranges[best_point_index]) < 1:
                speed -= 1.0
            if self.print_speeds:
              print("base speed: ", speed)
        
        speed = min(speed, 3.0)     

        return speed

    def preprocess_lidar_scan(self, scan_msg):
        """
        Preprocesses LiDAR scan data by truncating the scan to a limited angle, filtering out NaNs,
        and clamping distances to a maximum value.
        
        Preconditions:
          - scan_msg is a valid LaserScan message containing ranges, angle_min, angle_max.
        
        Postconditions:
          - Returns a list of filtered and smoothed range values.
        
        Rep invariants:
          - The returned list length corresponds to the truncated window minus the window used for smoothing.
          - No value in the returned list exceeds 10.0.
        """
        if not self.isTruncated:
            total_range = len(scan_msg.ranges)
            range_size = int((self.range_angle / (scan_msg.angle_max - scan_msg.angle_min)) * total_range)
            self.truncated_start_index = (total_range // 2) - (range_size // 2)
            self.truncated_end_index = (total_range // 2) + (range_size // 2)
            self.isTruncated= True

        filtered_ranges = []
        for i in range(self.truncated_start_index, self.truncated_end_index):
            if np.isnan(scan_msg.ranges[i]):
                filtered_ranges.append(0.0)
            elif scan_msg.ranges[i] > 10.0 or np.isinf(scan_msg.ranges[i]):
                filtered_ranges.append(10.0)
            else:
                filtered_ranges.append(scan_msg.ranges[i])
        
        smooth_ranges = []
        window = 5
        for i in range(window, len(filtered_ranges) - window):
            smooth_ranges.append(
                np.min(filtered_ranges[i - window:i + window])
            )
        return smooth_ranges

    def draw_safety_bubble(self, ranges, center_index):
        """
        Zeros out points within a safety bubble radius around the closest obstacle.
        
        Preconditions:
          - ranges is a list of float values representing LiDAR distances.
          - center_index is a valid index in ranges.
        
        Postconditions:
          - The value at center_index is set to 0.0.
          - Neighboring values within the distance (center_distance + bubble_radius) are also set to 0.0.
        """
        center_distance = ranges[center_index]
        ranges[center_index] = 0.0

        # Zero out points forward
        for i in range(center_index + 1, len(ranges)):
            if ranges[i] > center_distance + self.bubble_radius:
                break
            ranges[i] = 0.0

        # Zero out points backward
        for i in range(center_index - 1, -1, -1):
            if ranges[i] > center_distance + self.bubble_radius:
                break
            ranges[i] = 0.0

    def find_max_gap(self, ranges):
        """
        Finds the largest contiguous sequence of range values that exceed a minimum gap distance.
        
        Preconditions:
          - ranges is a list of floats representing preprocessed distances.
        
        Postconditions:
          - Returns a tuple (start_index, end_index) corresponding to the maximum gap interval.
        
        Rep invariant:
          - For all indices in the returned interval, each value is greater than min_gap_distance.
        """
        max_start, max_size = 0, 0
        current_start, current_size = 0, 0

        for i, value in enumerate(ranges):
            if value > self.min_gap_distance:
                if current_size == 0:
                    current_start = i
                current_size += 1
            else:
                if current_size > max_size:
                    max_start, max_size = current_start, current_size
                current_size = 0

        if current_size > max_size:
            max_start, max_size = current_start, current_size

        return max_start, max_start + max_size

    def find_best_point(self, start_index, end_index):
        """
        Selects the best point in the gap as the midpoint between start_index and end_index.
        
        Preconditions:
          - start_index and end_index are valid indices with start_index < end_index.
        
        Postconditions:
          - Returns an integer index representing the center of the gap.
        """
        return (start_index + end_index) // 2
    
    def pid_control(self, error):
        """
        Computes a steering angle using a PID controller.
        
        Preconditions:
          - error is a float representing the difference (in radians) between the desired and current angle.
        
        Postconditions:
          - Returns a steering angle in radians, clipped to ±90°.
        
        Rep invariant:
          - The PID integral and previous error are updated consistently with each call.
        """
        # Integral
        self.pid_integral += error * self.dt

        # Derivative
        derivative = (error - self.pid_prev_error) / self.dt
        self.pid_prev_error = error

        # PID formula
        angle = self.kp * error + self.ki * self.pid_integral + self.kd * derivative

        # Optional: clip final angle to avoid saturating
        max_steering = np.deg2rad(90)  # ±90° limit
        angle = np.clip(angle, -max_steering, max_steering)

        if self.print_angles:
          print("original angle: ", np.rad2deg(angle))

        if np.abs(angle) < np.deg2rad(4.5):
            angle = np.square(angle) * np.sign(angle) * 0.7
        elif np.abs(angle) < np.deg2rad(7):
            angle = np.square(angle) * np.sign(angle) * 1.0
        else:
            angle = angle * 0.75

        if np.abs(angle) > self.large_angle_threshold:
            angle = angle * 3.5

        if self.print_angles:
          print("processed angle: ", np.rad2deg(angle))
        
        return angle

    def safety_stop(self, ranges) -> bool:
        """
        Performs an emergency stop if any scan values near the midpoint indicate a dangerously close obstacle.
        
        Preconditions:
          - ranges is a list of LiDAR distances.
        
        Postconditions:
          - If any range in a small window around the midpoint is less than 0.3, the car is commanded to stop.
          - Returns True if an emergency stop was triggered; otherwise, returns False.
        """
        midpoint = len(ranges) // 2
        for i in range(midpoint-5, midpoint+5):
            if ranges[i] < 0.3:
                self.publish_control(0.0, 0.0)
                print("EMERGENCY STOP")
                self.stop = True
                return True
        return False
    
    def new_safety_stop(self, scan_msg, ranges, start_index, end_index) -> bool:
        """
        Performs additional safety checks based on the largest gap's average distance and width.
        
        Preconditions:
          - scan_msg is a valid LaserScan message.
          - ranges is a preprocessed list of LiDAR distances.
          - start_index and end_index define a contiguous gap.
        
        Postconditions:
          - If the average distance or width (converted to angle steps) falls below a threshold, triggers an emergency stop.
          - Returns True if an emergency stop was triggered; otherwise, returns False.
        """
        largest_gap_average_distance = np.max(ranges[start_index:end_index])
        largest_gap_size = end_index - start_index

        self.largest_gap_distance_history.append(largest_gap_average_distance)
        if len(self.largest_gap_distance_history) > self.largest_gap_distance_history_size:
            self.largest_gap_distance_history.pop(0)
        #print("largest_gap_distance_history: ", self.largest_gap_distance_history)
        if np.mean(self.largest_gap_distance_history) < self.stop_threshold:
            self.publish_control(0.0, 0.0)
            print("EMERGENCY STOP GAP DISTANCE")
            self.stop = True
            return True

        self.largest_gap_history.append(largest_gap_size)
        if len(self.largest_gap_history) > self.largest_gap_history_size:
            self.largest_gap_history.pop(0)
        #print("largest_gap_history: ", self.largest_gap_history)
        gap_num_threshold = np.deg2rad(30) // scan_msg.angle_increment
        if np.mean(self.largest_gap_history) < gap_num_threshold:
            self.publish_control(0.0, 0.0)
            print("EMERGENCY STOP GAP WIDTH")
            self.stop = True
            return True

        return False
    
    def check_laps(self) -> bool:
        """
        Checks if the lap count has exceeded the desired number of laps and stops the car if so.
        
        Preconditions:
          - self.num_laps is set appropriately (with -1 meaning indefinite laps).
        
        Postconditions:
          - If lap_count > num_laps (and num_laps != -1), stops the car and returns True.
          - Otherwise, returns False.
        """
        if self.num_laps != -1 and self.lap_count > self.num_laps:
            self.publish_control(0.0, 0.0)
            print("Lap limit reached. Stopping car.")
            self.stop = True
            return True
        return False

    def publish_control(self, angle, speed):
        """
        Publishes the control message with the given steering angle and speed.
        
        Preconditions:
          - angle is the computed steering angle (in radians).
          - speed is the desired speed (float).
        
        Postconditions:
          - Publishes an AckermannDriveStamped message.
          - Updates internal speed history and adjusts the safety stop threshold based on average speed.
        """
        new_msg = AckermannDriveStamped()
        new_msg.drive.speed = speed
        new_msg.drive.steering_angle = angle
        self.publisher_.publish(new_msg)
        self.speed = speed
        self.speed_history.append(speed)

        if len(self.speed_history) > self.speed_history_size:
            self.speed_history.pop(0)
        speed_average = np.mean(self.speed_history) 

        if self.print_safety_stop_thresholds:
          print("speed average: ", speed_average)

        if 0 <= speed_average <= 1.0:
            self.stop_threshold = 1.0
            if self.print_safety_stop_thresholds:
              print("slow stop_threshold: ", self.stop_threshold)
        elif 1.0 < speed_average <= 2.0:
            self.stop_threshold = 1.5
            if self.print_safety_stop_thresholds:
              print("medium stop_threshold: ", self.stop_threshold)
        elif 2.0 < speed_average <= 3.0:
            self.stop_threshold = 2.5
            if self.print_safety_stop_thresholds:
              print("fast stop_threshold: ", self.stop_threshold)
        elif 3.0 < speed_average <= 4.0:
            self.stop_threshold = 3.0
            if self.print_safety_stop_thresholds:
              print("faster stop_threshold: ", self.stop_threshold)
        else:
            self.stop_threshold = 3.5
            if self.print_safety_stop_thresholds:
              print("fastest stop_threshold: ", self.stop_threshold)

    def save_lidar_csv(self, lidar_data):
        """
        Saves the current LiDAR data to a CSV file with a timestamp.
        
        Preconditions:
          - lidar_data is a list of processed LiDAR range values.
        
        Postconditions:
          - Appends a new row to the CSV file at self.lidar_csv_path.
          - Writes a header row once if it hasn't been written already.
        """
        # Get the current time in seconds from the ROS clock
        timestamp = self.get_clock().now().nanoseconds / 1e9
        row = [timestamp] + lidar_data
        with open(self.lidar_csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write a header row if it hasn't been written yet
            if not self.csv_header_written:
                header = ["timestamp"] + [f"point_{i}" for i in range(len(lidar_data))]
                writer.writerow(header)
                self.csv_header_written = True
            writer.writerow(row)

    def scan_callback(self, scan_msg):
        """
        Callback for LaserScan messages. Processes the LiDAR data to compute a safe driving command,
        perform gap-following, and publish the control message.
        
        Preconditions:
          - scan_msg is a valid LaserScan message.
          - A reference image has been set (self.reference_set is True).
        
        Postconditions:
          - If no safety issues are detected, computes a new steering angle and speed, then publishes the control command.
          - Returns early if emergency stop conditions or lap limit conditions are met.
        """
        if not self.reference_set:
            return

        # Preprocess the scan
        filtered_ranges = self.preprocess_lidar_scan(scan_msg)

        # Save the lidar data to a CSV file
        # self.save_lidar_csv(filtered_ranges)

        # Safety stop
        if self.stop or self.safety_stop(scan_msg.ranges) or self.check_laps():
            return

        # Find the closest point
        closest_index = np.argmin(filtered_ranges)
        # closest_range = filtered_ranges[closest_index]

        # Zero out the bubble
        self.draw_safety_bubble(filtered_ranges, closest_index)

        # Find the largest gap
        start_index, end_index = self.find_max_gap(filtered_ranges)
            
        # Safety stop
        if self.new_safety_stop(scan_msg, filtered_ranges, start_index, end_index):
            return

        # Get the best point in the gap
        best_point_index = self.find_best_point(start_index, end_index)

        # Calculate the error
        midpoint = len(filtered_ranges) // 2
        error_index = best_point_index - midpoint
        error_angle = error_index * scan_msg.angle_increment

        # Calculate the steering angle
        steering_angle = self.pid_control(error_angle)

        # # Smooth the steering angle
        # self.steering_history.append(steering_angle)
        # if len(self.steering_history) > self.steering_history_size:
        #     self.steering_history.pop(0)
        # average_steering_angle = np.mean(self.steering_history)

        # Publish the drive message
        speed = self.get_speed(steering_angle, best_point_index, filtered_ranges)
        self.publish_control(steering_angle, speed)

    def process_image_for_comparison(self, image):
        """
        Processes an input image to create a binary template used for lap counting.
        The process includes:
          - Converting to grayscale.
          - Cropping the top portion of the image.
          - Masking specific regions to focus on the track.
          - Applying binary thresholding.
          - Using morphological closing to fill white spots.
        
        Preconditions:
          - image is a valid OpenCV image in BGR format.
        
        Postconditions:
          - Returns a processed binary image suitable for template matching.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Crop
        cropped_h = int(0.30 * h)  # Keep top 30%
        cropped_w = int(0.80 * w)  # Keep left 80%
        gray = gray[:cropped_h, :cropped_w]  # Apply cropping

        # Get new dimensions after cropping
        h, w = gray.shape[:2]

        # Define mask dimensions
        mask_height = int(0.70 * h)  # Keep top 30%
        mask_left = int(0.05 * w)  # Mask out left 5%
        mask_right = int(0.90 * w)  # Mask out right 10% (keeping left 90%)

        # Create mask and apply it
        masked = np.zeros_like(gray)
        masked[:mask_height, mask_left:mask_right] = gray[:mask_height, mask_left:mask_right]

        # Apply binary thresholding
        _, bw = cv2.threshold(masked, 127, 255, cv2.THRESH_BINARY)

        # Fill white spots using morphological closing
        kernel = np.ones((5, 5), np.uint8)
        inv_bw = cv2.bitwise_not(bw)
        inv_bw_closed = cv2.morphologyEx(inv_bw, cv2.MORPH_CLOSE, kernel)
        bw_filled = cv2.bitwise_not(inv_bw_closed)

        return bw_filled
        
    # def process_image(self, image):
    #     # Get image dimensions
    #     #height, width = image.shape

    #     # Prepare visualization
    #     if self.display_image:
            
    #         # Save the processed image
    #         os.makedirs(self.output_folder, exist_ok=True)  # Ensure the folder exists
    #         # filename = os.path.join(self.output_folder, f"processed_{int(time.time()*1000)}.png") # Save with unique timestamp
    #         filename = os.path.join(self.output_folder, "processed_image.png")  # Fixed filename
    #         cv2.imwrite(filename, image)

    #         # # Display the processed image
    #         # cv2.imshow('Image', image)
    
    def image_callback(self, msg):
        """
        Callback for camera images. Converts the ROS Image to OpenCV format and handles
        reference image storage and lap counting via template matching.
        
        Preconditions:
          - msg is a valid sensor_msgs/Image.
        
        Postconditions:
          - On the first image, sets and saves the reference image.
          - On subsequent images, processes and compares them to the reference image to detect laps.
          - Optionally displays a side-by-side comparison if display_image is True.
        """
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        
        # Optionally convert image to float32 if needed.
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # If the reference image is not set, process and save it, then return.
        if not self.reference_set:
            processed = self.process_image_for_comparison(image)
            self.processed_ref = processed
            self.reference_set = True
            # Save the original reference image (optional) and/or the processed version.
            ref_filename = os.path.join(self.output_folder, "reference_image.png")
            cv2.imwrite(ref_filename, image)
            ref_filename = os.path.join(self.output_folder, "reference_image_processed.png")
            cv2.imwrite(ref_filename, processed)
            print(f"Reference and processed image saved")
            return  # Do not perform lap comparison until reference is set.

        # Lap counting: if in ignoring period, check if it is over.
        if self.ignoring:
            elapsed = time.time() - self.ignoring_start_time
            if self.print_lap_timer:
                print(f"Ignoring further matches for {elapsed:.1f} seconds")
            if elapsed < self.ignoring_duration:
                cv2.waitKey(1)
                return
            else:
                self.ignoring = False

        # Process the current image for comparison.
        processed_current = self.process_image_for_comparison(image)

        # Perform template matching between processed current image and stored processed reference.
        result = cv2.matchTemplate(processed_current, self.processed_ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= self.detect_threshold:
            # Resize matched image to match reference image size and combine side by side
            matched_resized = cv2.resize(processed_current, (self.processed_ref.shape[1], self.processed_ref.shape[0]))
            combined = np.hstack((self.processed_ref, matched_resized))
            filename = os.path.join(self.output_folder, f"matched_image_{self.lap_count}.png") # Save with lap count
            cv2.imwrite(filename, combined)
            self.lap_count += 1
            print(f"Lap counted! Current lap count: {self.lap_count - 1}")
            self.ignoring = True
            self.ignoring_start_time = time.time()

        # (Optional) Display images side by side if display_image is True.
        if self.display_image:
            disp_height = 400

            # Resize original current image.
            h_current, w_current = image.shape[:2]
            ratio_current = w_current / h_current
            disp_width_current = int(disp_height * ratio_current)
            orig_current_resized = cv2.resize(image, (disp_width_current, disp_height))

            # Resize processed reference image.
            h_ref, w_ref = self.processed_ref.shape[:2]
            ratio_ref = w_ref / h_ref
            disp_width_ref = int(disp_height * ratio_ref)
            processed_ref_resized = cv2.resize(cv2.cvtColor(self.processed_ref, cv2.COLOR_GRAY2BGR),
                                                (disp_width_ref, disp_height))

            # Resize processed current image.
            processed_current_resized = cv2.resize(cv2.cvtColor(processed_current, cv2.COLOR_GRAY2BGR),
                                                    (disp_width_current, disp_height))

            side_by_side = np.hstack([orig_current_resized, processed_ref_resized, processed_current_resized])
            cv2.imshow("Side by Side", side_by_side)
            cv2.waitKey(1)
        
def main(args=None):
    rclpy.init(args=args)
    print('AutoDrive Initialized')
    auto_drive_node = AutoDrive()
    rclpy.spin(auto_drive_node)
    auto_drive_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()