#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Bool

import time

class AutoDriveNode(Node):
    """
    Node that handles:
      - LIDAR-based gap following
      - Speed control
      - 'Stop', 'Yield', 'Limit-X', or 'U-turn' sign actions
      - U-turn state machine
    """

    def __init__(self):
        super().__init__('auto_drive_node')

        # Topics
        self.scan_topic = '/scan'
        self.drive_topic = '/drive'
        self.odom_topic = '/odom'
        self.detected_sign_topic = '/detected_sign'
        self.enable_detection_topic = '/enable_detection'

        # Publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.enable_detection_pub = self.create_publisher(Bool, self.enable_detection_topic, 10)

        # Subscribers
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.detected_sign_sub = self.create_subscription(String, self.detected_sign_topic, self.detected_sign_callback, 10)

        # Driving parameters
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

        # Variables for truncated range
        self.isTruncated = False
        self.truncated_start_index = 0
        self.truncated_end_index = 0

        # PID parameters
        self.kp = 0.65
        self.kd = 0.15
        self.ki = 0.0

        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.dt = 0.05  # 20 Hz

        # Speed settings for different limit signs
        self.speed_20 = False
        self.speed_100 = False

        # Object actions
        self.trigger_labels = {"Stop", "Yield", "U-Turn", "Limit-20", "Limit-100"}
        self.object_timer_end = {}
        self.label_distance_threshold = {
            "Stop": 2.0,
            "Yield": 1.0,
            "U-Turn": 2.0,
            "Limit-20": 2.0,
            "Limit-100": 2.0
        }
        self.label_cooldowns = {
            "Stop": 10.0,
            "Yield": 20.0,
            "U-Turn": 10.0,
            "Limit-20": 10.0,
            "Limit-100": 10.0
        }
        self.last_triggered_time = {}

        # U-turn state machine
        self.first_time = False
        self.u_turn_active = False
        self.u_turn_state = None
        self.yaw_initial = None
        self.desired_distance = 0.4
        self.turn_phase = None
        self.phase_start_yaw = None
        self.phase_start_position = None

        # Additional U-turn PID
        self.ut_kp = 0.5
        self.ut_ki = 0.0001
        self.ut_kd = 0.3
        self.ut_integral = 0.0
        self.ut_prev_error = 0.0

        # Debugging
        self.print_speeds = False
        self.print_angles = False
        self.print_safety_stop_thresholds = False
        self.print_object_timers = True

        # Subscribe once for the entire node lifetime
        print("AutoDriveNode initialized.")

    ############
    # Sign / Object-Detection subscription
    ############
    def detected_sign_callback(self, msg: String):
        """
        Each message from object_detection_node is of the form "Label|Distance".
        We'll parse the label, parse the distance, then trigger or schedule the action.
        """
        data = msg.data
        # e.g. "Stop|1.52", "U-Turn|4.30", "Limit-20|2.10"
        parts = data.split('|')
        if len(parts) != 2:
            return  # malformed

        label = parts[0]
        try:
            distance_m = float(parts[1])
        except ValueError:
            distance_m = 999.9

        # We check if it's one of the known trigger labels
        if label in self.trigger_labels:
            self.update_object_timer(label, distance_m)

    def update_object_timer(self, label, distance):
        """
        We estimate time to reach the object based on average speed & distance minus threshold.
        Then we store it in object_timer_end so check_object_timers() can handle it at the correct time.
        """
        # Average speed from history (or default)
        if len(self.speed_history) > 0:
            avg_speed = np.mean(self.speed_history)
        else:
            avg_speed = self.speed

        threshold = self.label_distance_threshold.get(label, 0.0)
        effective_distance = max(distance - threshold, 0.0)

        if avg_speed > 0:
            time_to_reach = effective_distance / avg_speed
        else:
            time_to_reach = float('inf')

        current_time = self.get_clock().now().nanoseconds / 1e9
        new_end = current_time + time_to_reach

        if self.object_timer_end.get(label) is None:
          # Update the timer end time for this label
          self.object_timer_end[label] = new_end
          if self.print_object_timers:
              print(f"Detected {label}. Will trigger in {time_to_reach:.2f}s (dist={effective_distance:.2f}m).")
       
    def check_object_timers(self):
        """
        Periodically check if any label's timer has expired. Then trigger the appropriate action.
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        to_trigger = [lbl for lbl, end_t in self.object_timer_end.items() if current_time >= end_t]

        for lbl in to_trigger:
            self.trigger_object_action(lbl)
            del self.object_timer_end[lbl]

    def trigger_object_action(self, label):
        """
        Called when the timer for a certain sign has elapsed, meaning we've 'reached' it and must act.
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        cooldown = self.label_cooldowns.get(label, 10.0)  # default to 10.0 if not found
        last_time = self.last_triggered_time.get(label, 0.0)

        # Check if the label is still in cooldown
        if (current_time - last_time) < cooldown:
            if self.print_object_timers:
              print(f"Action for '{label}' is still in cooldown. Ignoring.")
            return  # Skip re-triggering the action

        # Update the last triggered time
        self.last_triggered_time[label] = current_time

        if label == "Stop":
            if self.print_object_timers:
              print("Triggering STOP action.")
            self.publish_control(0.0, 0.0)
            time.sleep(3)  # 3s stop
        elif label == "Yield":
            if self.print_object_timers:
                print("Triggering YIELD action (placeholder).")
        elif label == "U-Turn":
            if self.print_object_timers:
              print("Triggering U-TURN action.")
            # Only do it once
            if not self.first_time:
                self.first_time = True
                self.initiate_u_turn()
        elif label == "Limit-20":
            if self.print_object_timers:
              print("Switching speed limit to ~20.")
            self.speed_20 = True
            self.speed_100 = False
        elif label == "Limit-100":
            if self.print_object_timers:
              print("Switching speed limit to ~100.")
            self.speed_20 = False
            self.speed_100 = True

    ############
    # LIDAR & Driving Callbacks
    ############

    def scan_callback(self, scan_msg):
        """
        Main callback for gap-following logic, used when not in U-turn mode.
        """
        # If we're in the middle of a U-turn, handle that instead
        if self.u_turn_active:
            self.handle_u_turn(scan_msg)
            return
        
        # Preprocess the scan
        filtered_ranges = self.preprocess_lidar_scan(scan_msg)

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
            
        # # Safety stop
        # if self.new_safety_stop(scan_msg, filtered_ranges, start_index, end_index):
        #     return

        # Get the best point in the gap
        best_point_index = self.find_best_point(start_index, end_index)

        # Calculate the error
        midpoint = len(filtered_ranges) // 2
        error_index = best_point_index - midpoint
        error_angle = error_index * scan_msg.angle_increment

        # Calculate the steering angle
        steering_angle = self.pid_control(error_angle)

        # Publish the drive message
        speed = self.get_speed()
        self.publish_control(steering_angle, speed)

        # Check timers for sign actions
        self.check_object_timers()
    
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
        self.drive_pub.publish(new_msg)
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

    def get_speed(self):
        """
        Returns the speed based on sign triggers (20 or 100).
        """
        if self.speed_20:
            return 0.7
        elif self.speed_100:
            return 1.3
        else:
            return 1.0

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
            angle = angle * 0.6

        if np.abs(angle) > self.large_angle_threshold:
            angle = angle * 1.8
        
        if self.print_angles:
            print("processed angle: ", np.rad2deg(angle))

        return angle

    ############
    # Odometry & U-Turn
    ############

    def odom_callback(self, msg: Odometry):
        # Compute yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_position_x = msg.pose.pose.position.x
        self.current_position_y = msg.pose.pose.position.y

        self.speed_odom = msg.twist.twist.linear.x  # used by U-turn logic

    def initiate_u_turn(self):
        """Start the U-turn procedure. Let the detection node know to disable detections."""
        print("Initiating U-turn. Disabling object detection now.")
        self.u_turn_active = True
        # Publish enable_detection = False
        disable_msg = Bool()
        disable_msg.data = False
        self.enable_detection_pub.publish(disable_msg)

        # Switch to first U-turn state
        self.u_turn_state = "approach"
        self.yaw_initial = self.current_yaw

    def handle_u_turn(self, scan_msg):
        """
        Called inside scan_callback whenever self.u_turn_active = True.
        We run the U-turn state machine, then re-enable detection when done.
        """
        if self.u_turn_state == "approach":
            self.u_turn_approach(scan_msg)
        elif self.u_turn_state == "u_turn":
            self.u_turn_turn(scan_msg)
        elif self.u_turn_state == "three_point_turn":
            self.u_turn_three_point_turn(scan_msg)
        elif self.u_turn_state == "wall_follow":
            print("U-turn maneuver done. Re-enabling object detection.")
            self.u_turn_active = False
            self.turn_phase = 1
            self.u_turn_state = None
            enable_msg = Bool()
            enable_msg.data = True
            self.enable_detection_pub.publish(enable_msg)
        #     self.u_turn_wall_follow(scan_msg)

    def u_turn_approach(self, scan_msg: LaserScan):
        # Example approach logic ...
        right_distance = self.get_range(scan_msg, -90)
        print(f"U-turn approach. Right distance: {right_distance:.2f}")
        # Just an example of controlling steering ...
        error = self.get_ut_error(scan_msg)
        angle = self.ut_pid_control(error)
        self.publish_control(angle, 0.5)

        # If close enough to the wall, proceed
        if right_distance < self.desired_distance:
            print("Approach done. Checking track width.")
            # check left side
            left_distance = self.get_range(scan_msg, 90)
            if left_distance < 2.0:
                print("Narrow track -> three-point turn.")
                self.u_turn_state = "three_point_turn"
                self.turn_phase = 1
                self.phase_start_yaw = self.current_yaw
            else:
                self.u_turn_state = "u_turn"

    def u_turn_turn(self, scan_msg: LaserScan):
        left_dist = self.get_range(scan_msg, 90)
        print(f"U-turn in progress. Left dist: {left_dist:.2f}")

        # Simple steering
        self.publish_control(math.radians(30), 0.5)

        # Check if we've turned enough (~180 deg)
        yaw_diff = self.normalize_angle(self.current_yaw - self.yaw_initial)
        if abs(yaw_diff) >= math.pi - 0.2:
            print("U-turn turn complete. Transition to wall_follow.")
            self.u_turn_state = "wall_follow"

    def u_turn_three_point_turn(self, scan_msg: LaserScan):
        # Example for multi-phase reversing, etc.
        if self.turn_phase == 1:
            # Phase 1: forward left
            self.publish_control(1.0, 0.5)  # steer left
            yaw_diff = abs(self.normalize_angle(self.current_yaw - self.phase_start_yaw))
            if yaw_diff >= math.radians(60):
                print("Three-point turn: Phase 1 done.")
                self.turn_phase = 2
                self.phase_start_yaw = self.current_yaw
                self.phase_start_position = (self.current_position_x, self.current_position_y)
        elif self.turn_phase == 2:
            # Phase 2: reverse right
            self.publish_control(-math.radians(30), -0.5)
            if self.phase_start_position is not None:
                dx = self.current_position_x - self.phase_start_position[0]
                dy = self.current_position_y - self.phase_start_position[1]
                rev_dist = math.sqrt(dx*dx + dy*dy)
                if rev_dist >= 0.25:
                    print("Three-point turn: Phase 2 done.")
                    self.turn_phase = 3
                    self.phase_start_yaw = self.current_yaw
        elif self.turn_phase == 3:
            # Phase 3: final alignment
            # yaw_error = 0.0 - self.current_yaw
            # print("yaw_error", yaw_error)
            # self.publish_control(-0.4 * yaw_erro, 0.5)
            # if abs(yaw_error) >= 5.0:
            #     print("Three-point turn complete. Switching to wall_follow.")
            #     self.u_turn_state = "wall_follow
            self.publish_control(math.radians(30), 0.5)
            if self.phase_start_position is not None:
                dx = self.current_position_x - self.phase_start_position[0]
                dy = self.current_position_y - self.phase_start_position[1]
                rev_dist = math.sqrt(dx*dx + dy*dy)
                if rev_dist >= 0.4:
                    print("Three-point turn complete. Switching to wall_follow.")
                    self.u_turn_state = "wall_follow"

    def u_turn_wall_follow(self, scan_msg: LaserScan):
        # After finishing the turn, we do short wall follow, then re-enable detection
        error = self.get_ut_error(scan_msg)
        angle = self.ut_pid_control(error)
        self.publish_control(angle, 0.5)

        right_distance = self.get_range(scan_msg, -90)
        if right_distance >= self.desired_distance + 0.2:
            print("U-turn maneuver done. Re-enabling object detection.")
            self.u_turn_active = False
            self.turn_phase = 1
            self.u_turn_state = None
            enable_msg = Bool()
            enable_msg.data = True
            self.enable_detection_pub.publish(enable_msg)

    ############
    # U-turn Helpers
    ############
    def get_range(self, scan_msg, angle_deg):
        angle_rad = math.radians(angle_deg)
        if angle_rad < scan_msg.angle_min or angle_rad > scan_msg.angle_max:
            return float('inf')
        idx = int((angle_rad - scan_msg.angle_min) / scan_msg.angle_increment)
        if idx < 0 or idx >= len(scan_msg.ranges):
            return float('inf')
        return scan_msg.ranges[idx]

    def get_ut_error(self, scan_msg):
        # simplistic example
        a = self.get_range(scan_msg, -45)
        b = self.get_range(scan_msg, -90)
        theta = math.radians(45)
        alpha = math.atan2(a*math.cos(theta) - b, a*math.sin(theta))
        D_t = b * math.cos(alpha)
        L = 0.9 + (self.speed_odom * 0.11)
        D_t1 = D_t + L * math.sin(alpha)
        return self.desired_distance - D_t1

    def ut_pid_control(self, error):
        self.ut_integral += error
        derivative = error - self.ut_prev_error
        self.ut_prev_error = error
        return self.ut_kp*error + self.ut_ki*self.ut_integral + self.ut_kd*derivative

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2*math.pi
        while angle < -math.pi:
            angle += 2*math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = AutoDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()