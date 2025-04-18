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
from ultralytics import YOLO 

class AutoDrive(Node):
    """ 
    Implement auto driving on the car.
    """
    def __init__(self):
        super().__init__('auto_drive_node')

        lidarscan_topic = '/scan'
        camera_topic = '/camera/camera/color/image_raw'
        depth_camera_topic = '/camera/camera/depth/image_rect_raw'
        drive_topic = '/drive'

        # Create subscribers and publishers
        self.publisher_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.image_subscription = self.create_subscription(Image, camera_topic, self.image_callback, 10)
        self.image_subscription # prevent unused variable warning
        self.depth_image_subscription = self.create_subscription(Image, depth_camera_topic, self.depth_image_callback, 10)
        self.depth_image_subscription # prevent unused variable warning
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

        self.yolo_model_path = "yolo11s_model/model.pt"  # YOLO model path
        self.confidence_threshold = 0.5
        self.yolo_model = YOLO(self.yolo_model_path)
        self.labels = self.yolo_model.names
        self.bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
       
        # Store latest images from callbacks
        self.latest_rgb = None
        self.latest_depth = None

        self.trigger_labels = {"Stop", "Yield", "U-Turn", "Limit-20", "Limit-100"}
        # Timer dictionary for object-based actions:
        # Keys: label (e.g., "Stop") ; Value: timer end time (in seconds)
        self.object_timer_end = {}
        # Distance threshold (in meters) for each label
        # When calculating time to reach, effective distance = avg_depth - threshold.
        self.label_distance_threshold = {
            "Stop": 0.5,
            "Yield": 1.0,
            "U-Turn": 1.0,
            "Limit-20": 2.0,
            "Limit-100": 2.0
        }

        # Debugging
        self.print_speeds = False
        self.print_angles = False
        self.print_safety_stop_thresholds = False
        self.print_lap_timer = False
        self.print_object_detection = True
        self.print_object_timers = True

    def get_speed(self, angle, best_point_index, ranges):
        """
        Sets the speed of the car based on the steering angle.
        """
        # if 0 <= abs(np.rad2deg(angle)) <= 2:
        #     speed = 1.5 + math.exp(0.04 * abs(ranges[best_point_index]))
        #     if self.print_speeds:
        #         print("speed 1: ", speed)
        # elif 2 < abs(np.rad2deg(angle)) <= 10:
        #     speed = 2.0
        #     if self.print_speeds:
        #         print("speed 2: ", speed)
        # elif 10 < abs(np.rad2deg(angle)) <= 15:
        #     speed = 1.8
        #     if self.print_speeds:
        #         print("speed 3: ", speed)
        # else:
        #     speed = 1.5
        #     if self.print_speeds:
        #         print("base speed: ", speed)

        # speed = min(speed, 3.0)

        speed = 0.7

        return speed

    def preprocess_lidar_scan(self, scan_msg):
        """
        Pre-process the LiDAR data to truncate the range, filter out NaNs and limit the range distance.
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
        Zero out points within the bubble radius around the closest point.
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
        Find the largest contiguous sequence of values greater than min_gap_distance.
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
        Get the best point in the range.
        """

        return (start_index + end_index) // 2
    
    def pid_control(self, error):
        """
        Compute the steering angle from error.
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
            angle = angle * 1.8
        
        if self.print_angles:
            print("processed angle: ", np.rad2deg(angle))

        return angle

    def safety_stop(self, ranges) -> bool:
        midpoint = len(ranges) // 2
        for i in range(midpoint-5, midpoint+5):
            if ranges[i] < 0.3:
                self.publish_control(0.0, 0.0)
                print("EMERGENCY STOP")
                self.stop = True
                return True
        return False
    
    def new_safety_stop(self, scan_msg, ranges, start_index, end_index) -> bool:
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

    def publish_control(self, angle, speed):
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
        Callback function for LaserScan messages.
        Implements gap follow and publishes the control message.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """

        if not self.reference_set:
            return

        # Preprocess the scan
        filtered_ranges = self.preprocess_lidar_scan(scan_msg)

        # Save the lidar data to a CSV file
        # self.save_lidar_csv(filtered_ranges)

        # Safety stop
        if self.stop or self.safety_stop(scan_msg.ranges):
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
    
    def colorize_depth(self, depth_image):
        norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        norm = np.uint8(norm)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def get_average_depth(self, depth_image, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = depth_image.shape

        # Ensure box is within image bounds
        x1 = np.clip(x1, 0, w - 1)
        x2 = np.clip(x2, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        y2 = np.clip(y2, 0, h - 1)

        region = depth_image[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        avg_mm = np.mean(region)
        avg_meters = avg_mm * 0.001  # Convert mm to meters

        return avg_meters

    def annotate_and_display(self, rgb_img, depth_img, detections):
        object_count = 0
        detection_results = []  # For printing and timer updates

        for detection in detections:
            xyxy = detection.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = detection.conf.item()
            classidx = int(detection.cls.item())
            classname = self.labels[classidx]

            if conf > self.confidence_threshold:
                color = self.bbox_colors[classidx % len(self.bbox_colors)]
                label = f"{classname}: {int(conf * 100)}%"

                # Draw on the images if display is enabled
                if self.display_image:
                    depth_colormap = self.colorize_depth(depth_img)
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color, 2)
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(y1, label_size[1] + 10)
                    cv2.rectangle(rgb_img, (x1, label_ymin - label_size[1] - 10), 
                                  (x1 + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
                    cv2.putText(rgb_img, label, (x1, label_ymin - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), color, 2)
                    avg_depth = self.get_average_depth(depth_img, (x1, y1, x2, y2))
                    cv2.putText(depth_colormap, f"Avg Depth: {avg_depth:.2f}", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Compute average depth for updating timers/printing
                avg_depth = self.get_average_depth(depth_img, (x1, y1, x2, y2))
                detection_results.append({
                    "class": classname,
                    "confidence": conf,
                    "avg_depth": avg_depth,
                    "bbox": (x1, y1, x2, y2)
                })
                object_count += 1

        if self.display_image:
            cv2.putText(rgb_img, f"Objects detected: {object_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            depth_resized = cv2.resize(depth_colormap, (rgb_img.shape[1], rgb_img.shape[0]))
            combined = np.hstack((rgb_img, depth_resized))
            os.makedirs(self.output_folder, exist_ok=True)
            # filename = os.path.join(self.output_folder, f"object_detect_image_{int(time.time()*1000)}.png") # Save with unique timestamp
            filename = os.path.join(self.output_folder, "object_detect_image.png")
            cv2.imwrite(filename, combined)
            cv2.waitKey(1)
        
        if self.print_object_detection:
            print("Objects detected:", object_count)
            for res in detection_results:
                print(f"{res['class']} - {int(res['confidence'] * 100)}% - Avg Depth: {res['avg_depth']:.2f} m - BBox: {res['bbox']}")

        return detection_results

    def update_object_timers(self, detection_results):
        # Use average speed from speed history (default to 0.7 if empty)
        if len(self.speed_history) > 0:
            avg_speed = np.mean(self.speed_history)
        else:
            avg_speed = 0.7
        current_time = self.get_clock().now().nanoseconds / 1e9

        for det in detection_results:
            label = det["class"]
            if label in self.trigger_labels:
                avg_depth = det["avg_depth"]  # in meters
                # Get the distance threshold for this label; default to 0 if not set.
                threshold = self.label_distance_threshold.get(label, 0)
                effective_distance = max(avg_depth - threshold, 0)
                time_to_reach = effective_distance / avg_speed if avg_speed > 0 else float('inf')
                new_timer_end = current_time + time_to_reach
                self.object_timer_end[label] = new_timer_end
                if self.print_object_timers:
                    print(f"Updated timer for {label}: will trigger in {time_to_reach:.2f} seconds (Effective Distance: {effective_distance:.2f} m).")

    def check_object_timers(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        labels_to_trigger = [label for label, end_time in self.object_timer_end.items() if current_time >= end_time]
        for label in labels_to_trigger:
            self.trigger_object_action(label)
            del self.object_timer_end[label]

    def trigger_object_action(self, label):
        # Placeholder function for when the timer expires
        if self.print_object_timers:
            print(f"Action triggered for {label}.")

    def process_and_annotate_images(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return
        rgb_img = self.latest_rgb.copy()
        depth_img = self.latest_depth.copy()
        results = self.yolo_model(rgb_img, verbose=False)
        detections = results[0].boxes
        detection_results = self.annotate_and_display(rgb_img, depth_img, detections)
        self.update_object_timers(detection_results)
        self.check_object_timers()

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        self.latest_rgb = image
        self.process_and_annotate_images()
        cv2.waitKey(1)
    
    def depth_image_callback(self, msg):     
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
        self.latest_depth = depth_image
        self.process_and_annotate_images()
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