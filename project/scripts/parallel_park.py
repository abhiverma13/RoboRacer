import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class ParallelNode(Node):
    """
    Node that handles:
      - Parallel park state machine
    """

    def __init__(self):
        super().__init__('u_turn_node')

        # Topics for LiDAR, drive commands, and odometry
        self.scan_topic = '/scan'
        self.drive_topic = '/drive'
        self.odom_topic = '/odom'

        # Create publisher and subscribers
        self.publisher_ = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

        # State machine:
        # "approach": approach the wall with a gap of 0.5 m,
        # "u_turn": execute the U-turn,
        # "three_point_turn": perform a multi-step turn if the track is too narrow,
        # "wall_follow": resume wall following indefinitely.
        self.state = "approach"

        # Desired distance for approach; later updated for wall following
        self.desired_distance = 0.3  # meters

        # PID gains for wall following (used in approach and wall_follow states)
        self.kp = 0.5
        self.ki = 0.0001
        self.kd = 0.3
        self.integral = 0.0
        self.prev_error = 0.0

        # Current speed, yaw, and position (from odometry)
        self.speed = 0.0
        self.current_yaw = 0.0
        self.current_position_x = 0.0
        self.current_position_y = 0.0
        self.yaw_initial = None  # recorded at start of U-turn

        # Desired lane heading after turning (radians; 0 means straight ahead)
        self.desired_heading = 0.0

        # Attributes for three-point turn
        self.turn_phase = None  # will be 1, 2, or 3 during a three-point turn
        self.phase_start_yaw = None
        self.phase_start_position = None  # for reverse distance measurement

    def scan_callback(self, scan_msg):
        """
        Callback for processing LiDAR scans. Delegates to different state handlers based on current mode.

        Preconditions:
            - scan_msg is a valid LaserScan message.

        Postconditions:
            - The appropriate state_* function is called based on current self.state.

        Rep invariant:
            - self.state is one of 'approach', 'parallel_park', 'wall_follow', or 'avoid_box'.
        """
        if self.state == "approach":
            self.state_approach(scan_msg)
        elif self.state == "parallel_park":
            self.state_parallel_park(scan_msg)
            if self.turn_phase is None:
                self.turn_phase = 1
                self.phase_start_yaw = self.current_yaw  # Initialize the phase start yaw
        elif self.state == "wall_follow":
            self.state_wall_follow(scan_msg)
        elif self.state == "avoid_box":
            self.state_avoid_box(scan_msg)

    def odom_callback(self, msg):
        """
        Updates yaw, speed, and position using Odometry message.

        Preconditions:
            - msg is a valid Odometry message.

        Postconditions:
            - self.speed, self.current_yaw, self.current_position_x, and self.current_position_y are updated.

        Rep invariant:
            - Orientation is converted from quaternion to yaw correctly.
        """
        self.speed = msg.twist.twist.linear.x
        q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(q)
        self.current_position_x = msg.pose.pose.position.x
        self.current_position_y = msg.pose.pose.position.y

    def quaternion_to_yaw(self, q):
        """
        Converts quaternion orientation to yaw angle (radians).

        Preconditions:
            - q is a geometry_msgs.msg.Quaternion.

        Postconditions:
            - Returns float angle in radians.

        Rep invariant:
            - Output lies in range [-π, π].
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_range(self, scan_msg, angle_deg):
        """
        Retrieves range measurement from a specific angle.

        Preconditions:
            - scan_msg is a valid LaserScan message.
            - angle_deg is within the field of view.

        Postconditions:
            - Returns float distance or inf if out of bounds.

        Rep invariant:
            - Index is clipped within bounds of scan_msg.ranges.
        """
        angle_rad = np.deg2rad(angle_deg)
        if angle_rad < scan_msg.angle_min or angle_rad > scan_msg.angle_max:
            self.get_logger().warn("Angle out of bounds!")
            return float('inf')
        index = int((angle_rad - scan_msg.angle_min) / scan_msg.angle_increment)
        if index < 0 or index >= len(scan_msg.ranges):
            return float('inf')
        return scan_msg.ranges[index]

    def get_error(self, scan_msg):
        """
        Computes lateral error using LiDAR points at -45° and -90°.

        Preconditions:
            - scan_msg contains valid range data.

        Postconditions:
            - Returns signed float representing deviation from desired wall distance.

        Rep invariant:
            - Uses projection with velocity compensation.
        """
        a = self.get_range(scan_msg, -45.0)
        b = self.get_range(scan_msg, -90.0)
        theta = np.deg2rad(45.0)
        alpha = np.arctan((a * np.cos(theta) - b) / (a * np.sin(theta)))
        D_t = b * np.cos(alpha)
        L = 0.9 + (self.speed * 0.11)
        D_t1 = D_t + L * np.sin(alpha)
        return self.desired_distance - D_t1

    def pid_control(self, error):
        """
        Basic PID controller for computing steering angle.

        Preconditions:
            - error is a float deviation from desired position.

        Postconditions:
            - Returns float control output.

        Rep invariant:
            - Integral and derivative terms are updated per cycle.
        """
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def speed_control(self, angle):
        """
        Determines base speed (constant in this case).

        Preconditions:
            - angle is a float.

        Postconditions:
            - Returns float value representing speed.

        Rep invariant:
            - Always returns 1.0.
        """
        return 1.0
    
    def state_approach(self, scan_msg):
        """
        Approaches a wall to desired distance before entering box detection.

        Preconditions:
            - scan_msg is a valid LaserScan.

        Postconditions:
            - Publishes drive commands, and transitions to 'avoid_box' if wall is close.

        Rep invariant:
            - Uses PID and LiDAR-based control.
        """
        error = self.get_error(scan_msg)
        steering_angle = self.pid_control(error)
        speed = self.speed_control(steering_angle)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher_.publish(drive_msg)

        right_distance = self.get_range(scan_msg, -90.0)
        self.get_logger().info("Right distance (approach): {:.2f}".format(right_distance))
        if right_distance < self.desired_distance:
            self.get_logger().info("Approach complete. Finding box next")
            self.yaw_initial = self.current_yaw
            distance = self.get_range(scan_msg, 0.0)
            
            if distance < 1.0:
                self.get_logger().info("Approach complete. Switching to avoid_box state.")
                self.yaw_initial = self.current_yaw
                # Transition to avoid_box state
                self.state = "avoid_box"
                # Initialize phase variables for avoid_box
                self.turn_phase = 1
                self.phase_start_yaw = self.current_yaw
                self.phase_start_position = (self.current_position_x, self.current_position_y)

    def state_avoid_box(self, scan_msg):
        """
        Executes a 3-phase maneuver to bypass box and initiate parking.
        - Phase 1: Turn wheels left by 45° and go straight for 0.4 meters.
        - Phase 2: Turn wheels right by 45° and go straight for 0.6 meters.
        - Phase 3: Switch to state_parallel_park.

        Preconditions:
            - scan_msg is a valid LaserScan.

        Postconditions:
            - Transitions from 'avoid_box' to 'parallel_park' state.

        Rep invariant:
            - State machine proceeds through 3 distinct phases.
        """
        # Ensure phase variables are initialized
        if self.turn_phase is None:
            self.turn_phase = 1
            self.phase_start_yaw = self.current_yaw
            self.phase_start_position = (self.current_position_x, self.current_position_y)

        if self.turn_phase == 1:
            drive_msg = AckermannDriveStamped()
            # Turn wheels left by 45° (in radians)
            drive_msg.drive.steering_angle = np.deg2rad(45)
            drive_msg.drive.speed = 0.5
            self.publisher_.publish(drive_msg)
            # Calculate distance traveled in phase 1
            dx = self.current_position_x - self.phase_start_position[0]
            dy = self.current_position_y - self.phase_start_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            self.get_logger().info("Avoid Box Phase 1: distance = {:.2f}".format(distance))
            if distance >= 0.7:
                self.get_logger().info("Avoid Box: Phase 1 complete.")
                self.turn_phase = 2
                # Reset the starting position for phase 2
                self.phase_start_position = (self.current_position_x, self.current_position_y)
        elif self.turn_phase == 2:
            drive_msg = AckermannDriveStamped()
            # Turn wheels right by 45° (negative value)
            drive_msg.drive.steering_angle = -np.deg2rad(45)
            drive_msg.drive.speed = 0.5
            self.publisher_.publish(drive_msg)
            dx = self.current_position_x - self.phase_start_position[0]
            dy = self.current_position_y - self.phase_start_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            self.get_logger().info("Avoid Box Phase 2: distance = {:.2f}".format(distance))
            if distance >= 0.46:
                self.get_logger().info("Avoid Box: Phase 2 complete.")
                self.turn_phase = 3
        elif self.turn_phase == 3:
            self.get_logger().info("Avoid Box: Phase 3 complete. Switching to parallel park state.")
            drive_msg = AckermannDriveStamped()
            # Turn wheels right by 45° (negative value)
            drive_msg.drive.steering_angle = np.deg2rad(0)
            drive_msg.drive.speed = 0.5
            self.publisher_.publish(drive_msg)
            dx = self.current_position_x - self.phase_start_position[0]
            dy = self.current_position_y - self.phase_start_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            self.get_logger().info("Avoid Box Phase 3: distance = {:.2f}".format(distance))
            if distance >= 1.55:
                self.get_logger().info("Avoid Box: Phase 3 complete.")
                self.turn_phase = None
                self.state = "parallel_park"


    def state_parallel_park(self, scan_msg):
        """
        Executes a parallel parking maneuver with 3 turning phases.
        - Phase 1: Reverse with wheels right until 45 degree turn
        - Phase 2: Reverse with maximum right steering until a specified reverse distance is reached.
        - Phase 3: Forward with corrective steering until heading aligns.

        Preconditions:
            - scan_msg is a valid LaserScan.

        Postconditions:
            - Adjusts angle and speed in sequence to reverse park.

        Rep invariant:
            - self.turn_phase controls sub-state logic.
        """
        if self.turn_phase == 1:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = -np.deg2rad(30)  # maximum right
            drive_msg.drive.speed = -0.6
            self.publisher_.publish(drive_msg)
            print(f"phase 1 turn left: {self.current_yaw} - {self.phase_start_yaw}")
            if abs(self.normalize_angle(self.current_yaw - self.phase_start_yaw)) >= np.deg2rad(30):  # 45°
                self.get_logger().info("Three-point turn: Phase 1 complete.")
                self.turn_phase = 2
                self.phase_start_yaw = self.current_yaw
                # Store position at start of reverse phase
                self.phase_start_position = (self.current_position_x, self.current_position_y)
        elif self.turn_phase == 2:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0  # no turn
            drive_msg.drive.speed = -0.5                      # reversing
            self.publisher_.publish(drive_msg)
            # Compute reverse distance from phase_start_position
            if self.phase_start_position is not None:
                dx = self.current_position_x - self.phase_start_position[0]
                dy = self.current_position_y - self.phase_start_position[1]
                reverse_distance = math.sqrt(dx*dx + dy*dy)
                self.get_logger().info("Three-point turn: Reverse distance = {:.2f}".format(reverse_distance))
                if reverse_distance >= 0.1:  # threshold for reverse distance; adjust as needed
                    self.get_logger().info("Three-point turn: Phase 2 complete.")
                    self.turn_phase = 3
                    self.phase_start_yaw = self.current_yaw
        elif self.turn_phase == 3:
            print("Phase 3: Aligning heading")
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = -0.4
            drive_msg.drive.steering_angle = np.deg2rad(30) 
            # Use a lower gain for a gentler correction
            self.publisher_.publish(drive_msg)
            if abs(self.normalize_angle(self.current_yaw - self.phase_start_yaw)) >= np.deg2rad(45):
                self.get_logger().info("Three-point turn complete. Resuming wall follow.")
                self.turn_phase = 4
                self.phase_start_yaw = self.current_yaw
        elif self.turn_phase == 4:
            print("Phase 4: Resuming")
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0  # no turn
            drive_msg.drive.speed = 0.0
            self.publisher_.publish(drive_msg)
             

    def state_wall_follow(self, scan_msg):
        """
        Wall following state after parking is complete.

        Preconditions:
            - scan_msg is a valid LaserScan.

        Postconditions:
            - Maintains wall-following distance using PID steering.

        Rep invariant:
            - Average of left and right distances used as target.
        """
        self.desired_distance = (self.get_range(scan_msg, -90.0) + self.get_range(scan_msg, 90.0))/2
        error = self.get_error(scan_msg)
        steering_angle = self.pid_control(error)
        speed = self.speed_control(steering_angle)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher_.publish(drive_msg)
        right_distance = self.get_range(scan_msg, -90.0)
        self.get_logger().info("Right distance (wall follow): {:.2f}".format(right_distance))
    
    def normalize_angle(self, angle):
        """
        Normalizes angle to [-π, π].

        Preconditions:
            - angle is a float in radians.

        Postconditions:
            - Returns wrapped angle.

        Rep invariant:
            - Output is within range [-π, π].
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = ParallelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
