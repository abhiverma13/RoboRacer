#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math

import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class AutoDrive(Node):
    """ 
    Implement auto driving on the car.
    """
    def __init__(self):
        super().__init__('auto_drive_node')

        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # Create subscribers and publishers
        self.publisher_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.scan_subscription = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.scan_subscription  # prevent unused variable warning

        # Configuration Parameters
        self.speed = 0.0
        self.bubble_radius = 0.35
        self.range_angle = np.deg2rad(110)
        #self.steering_factor = 0.8

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
    
    def get_speed(self, angle, best_point_index, ranges):
        """
        Sets the speed of the car based on the steering angle.
        """
        if 0 <= abs(np.rad2deg(angle)) <= 10:
            print("CURRENT DETECTED RANGE: " + str(ranges[best_point_index]))
            speed = 0.2 + math.exp(0.04 * abs(ranges[best_point_index]))
            print("CURRENT SPEED: " + str(speed))
        elif 10 < abs(np.rad2deg(angle)) <= 20:
            speed = 1.0
        elif 20 < abs(np.rad2deg(angle)) <= 30:
            speed = 0.7
        else:
            speed = 0.5
        self.speed = speed
        return speed

    # def get_speed(self, angle):
    #     """
    #     Sets the speed of the car based on the steering angle.
    #     """
    #     if 0 <= abs(np.rad2deg(angle)) <= 5:
    #         speed = 3.5
    #     elif 5 < abs(np.rad2deg(angle)) <= 10:
    #         speed = 1.5
    #     elif 10 < abs(np.rad2deg(angle)) <= 20:
    #         speed = 1.0
    #     elif 20 < abs(np.rad2deg(angle)) <= 30:
    #         speed = 0.75
    #     else:
    #         speed = 0.5
    #     self.speed = speed
    #     return speed

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
            elif scan_msg.ranges[i] > scan_msg.range_max or np.isinf(scan_msg.ranges[i]):
                filtered_ranges.append(scan_msg.range_max)
            else:
                filtered_ranges.append(scan_msg.ranges[i])
        
        smooth_ranges = []
        window = 5
        for i in range(window, len(filtered_ranges) - window):
            smooth_ranges.append(
                np.mean(filtered_ranges[i - window:i + window])
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
        Find the largest contiguous sequence of non-zero values.
        """
        max_start, max_size = 0, 0
        current_start, current_size = 0, 0

        for i, value in enumerate(ranges):
            if value > 0.1:
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

        return angle

    # def get_steering_angle(self, scan_msg, best_point_index, closest_value):
    #     """
    #     Calculate the steering angle based on the best point index and reactivity factor.
    #     """
    #     best_point_index += self.truncated_start_index
    #     midpoint = len(scan_msg.ranges) // 2
    #     angle = scan_msg.angle_increment * (best_point_index - midpoint)

    #     # Adjust steering angle based on reactivity
    #     return np.clip((angle * self.steering_factor) / closest_value, np.deg2rad(-90), np.deg2rad(90))

    def safety_stop(self, scan_msg) -> bool:
            
            threshold = 1.2 # Tuned this value with some trial and error, seems conservative enough
            # Yet still enables car to pass through narrow hallway in the back 
            ranges = scan_msg.ranges
            num = len(ranges)
            scan_time = scan_msg.scan_time 
            angle_min = scan_msg.angle_min
            angle_max = scan_msg.angle_max
            angle_increment = scan_msg.angle_increment
            v = self.speed
            for i in range(540-5, 540+5): # this only considers a small cone in front of the car
                angle = angle_min + i*angle_increment # Compute angle for i'th element of ranges[]
                v_long = v * np.cos(angle) # Compute longitudinal v component i.e. in direction of this LiDAR beam
                if v_long <= 0:
                    v_long = 0 # Compute {v_long}+ aka max(0,v_long), to capture risk of distance getting smaller
                else:
                    iTTC = ranges[i] / v_long # Compute time to collision, using formula provided
                if v_long > 0 and iTTC < threshold or ranges[i] < 0.5: # Check if distance is decreasing, and doing so too fast
                    new_msg = AckermannDriveStamped()
                    new_msg.drive.speed = 0.0
                    new_msg.drive.steering_angle = 0.0
                    self.publisher_.publish(new_msg)
                    prt = "EMERGENCY STOP: iTTC = " + str(iTTC)
                    self.get_logger().info(prt)
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
        
        # if self.safety_stop(scan_msg):
        #     return

        # Preprocess the scan
        filtered_ranges = self.preprocess_lidar_scan(scan_msg)

        # Find the closest point
        closest_index = np.argmin(filtered_ranges)
        # closest_range = filtered_ranges[closest_index]

        # Zero out the bubble
        self.draw_safety_bubble(filtered_ranges, closest_index)

        # Find the largest gap
        start_index, end_index = self.find_max_gap(filtered_ranges)

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
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = self.get_speed(steering_angle, best_point_index, filtered_ranges)

        self.publisher_.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    print('AutoDrive Initialized')
    auto_drive_node = AutoDrive()
    rclpy.spin(auto_drive_node)
    auto_drive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
