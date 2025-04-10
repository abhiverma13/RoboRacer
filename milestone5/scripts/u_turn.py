import rclpy
from rclpy.node import Node
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

class UTurnNode(Node):
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
        self.desired_distance = 0.3 # meters

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
        if self.state == "approach":
            self.state_approach(scan_msg)
        elif self.state == "u_turn":
            self.state_u_turn(scan_msg)
        elif self.state == "three_point_turn":
            self.state_three_point_turn(scan_msg)
        elif self.state == "wall_follow":
            self.state_wall_follow(scan_msg)

    def odom_callback(self, msg):
        self.speed = msg.twist.twist.linear.x
        q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_yaw(q)
        self.current_position_x = msg.pose.pose.position.x
        self.current_position_y = msg.pose.pose.position.y

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_range(self, scan_msg, angle_deg):
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
        Compute error based on the distance from the right wall.
        Uses LiDAR measurements at -45째 (diagonal) and -90째 (directly right).
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
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def speed_control(self, angle):
        return 0.5

    def state_approach(self, scan_msg):
        error = self.get_error(scan_msg)
        steering_angle = self.pid_control(error)
        speed = self.speed_control(steering_angle)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher_.publish(drive_msg)

        right_distance = self.get_range(scan_msg, -70.0)
        self.get_logger().info("Right distance (approach): {:.2f}".format(right_distance))
        if right_distance < self.desired_distance:
            self.get_logger().info("Approach complete. Initiating turn maneuver.")
            self.yaw_initial = self.current_yaw
            left_distance = self.get_range(scan_msg, 70.0)
            if left_distance < 2.0:  # narrow track detected
                self.get_logger().info("Track too narrow; initiating three-point turn.")
                self.state = "three_point_turn"
                self.turn_phase = 1
                self.phase_start_yaw = self.current_yaw
                self.phase_start_position = None
            else:
                self.state = "u_turn"

    def state_u_turn(self, scan_msg):
        left_distance = self.get_range(scan_msg, 90.0)
        self.get_logger().info("Left distance (u_turn): {:.2f}".format(left_distance))
        
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = np.deg2rad(30)
        drive_msg.drive.speed = 0.7
        self.publisher_.publish(drive_msg)

        if self.yaw_initial is not None:
            yaw_diff = self.normalize_angle(self.current_yaw - self.yaw_initial)
            if abs(yaw_diff) >= math.pi - 0.2:
                self.get_logger().info("U-turn completed. Transitioning to wall follow.")
                self.state = "wall_follow"

    def state_three_point_turn(self, scan_msg):
        """
        Three-point turn sequence using three phases:
         - Phase 1: Forward turn with maximum left steering until a 45째 yaw change.
         - Phase 2: Reverse with maximum right steering until a specified reverse distance is reached.
         - Phase 3: Forward with corrective steering until heading aligns.
        """
        if self.turn_phase == 1:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = np.deg2rad(30)  # maximum left
            drive_msg.drive.speed = 0.7
            self.publisher_.publish(drive_msg)
            print(f"phase 1 turn left: {self.current_yaw} - {self.phase_start_yaw}")
            if abs(self.normalize_angle(self.current_yaw - self.phase_start_yaw)) >= np.deg2rad(85):  # 45째
                self.get_logger().info("Three-point turn: Phase 1 complete.")
                self.turn_phase = 2
                self.phase_start_yaw = self.current_yaw
                # Store position at start of reverse phase
                self.phase_start_position = (self.current_position_x, self.current_position_y)
        elif self.turn_phase == 2:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = -np.deg2rad(30)  # maximum right
            drive_msg.drive.speed = -0.6           # reversing
            self.publisher_.publish(drive_msg)
            # Compute reverse distance from phase_start_position
            if self.phase_start_position is not None:
                dx = self.current_position_x - self.phase_start_position[0]
                dy = self.current_position_y - self.phase_start_position[1]
                reverse_distance = math.sqrt(dx*dx + dy*dy)
                self.get_logger().info("Three-point turn: Reverse distance = {:.2f}".format(reverse_distance))
                if reverse_distance >= 0.3:  # threshold for reverse distance; adjust as needed
                    self.get_logger().info("Three-point turn: Phase 2 complete.")
                    self.turn_phase = 3
                    self.phase_start_yaw = self.current_yaw
        elif self.turn_phase == 3:
            print("Phase 3: Aligning heading")
            drive_msg = AckermannDriveStamped()
            yaw_error = self.desired_heading - self.current_yaw
            drive_msg.drive.speed = 0.7
            drive_msg.drive.steering_angle = np.deg2rad(30) 
            # Use a lower gain for a gentler correction
            self.publisher_.publish(drive_msg)
            self.get_logger().info(f"Phase 3: yaw_error = {yaw_error:.2f}")
            if abs(self.normalize_angle(self.current_yaw - self.phase_start_yaw)) >= np.deg2rad(30):
                self.get_logger().info("Three-point turn complete. Resuming wall follow.")
                self.integral = 0.0
                self.prev_error = 0.0
                self.state = "wall_follow"


    def state_wall_follow(self, scan_msg):
        self.desired_distance = (self.get_range(scan_msg, -70.0) + self.get_range(scan_msg, 70.0))/2
        error = self.get_error(scan_msg)
        steering_angle = self.pid_control(error)
        speed = self.speed_control(steering_angle)
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.publisher_.publish(drive_msg)
        right_distance = self.get_range(scan_msg, -70.0)
        self.get_logger().info("Right distance (wall follow): {:.2f}".format(right_distance))
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    node = UTurnNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()