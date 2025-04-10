#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from ackermann_msgs.msg import AckermannDriveStamped

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import time

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # get camera images from the simulator
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # Publisher for drive commands
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.bridge = CvBridge()

        self.scale_down_width_factor = 1.0
        self.scale_down_height_factor = 1.0
        self.mask_height = 310
        self.mask_vertices = [np.array([[0,480], [0, 340], [320, self.mask_height], [640, 340], [640, 480]], dtype=np.int32)]

        self.thresh_value = 50

        self.erode_kernel = np.ones((5,5), np.uint8)

        self.lookahead_distance = 0.2

        self.prev_mid_pts = {}

        # PID controller parameters
        self.pid_kp = 0.005
        self.pid_ki = 0.0001
        self.pid_kd = 0.001
        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.last_time = time.time()

    def image_callback(self, msg):
        try:
            # ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # wall detection and lane centering
        steering_angle, throttle = self.process_image(cv_image)

        # publish control commands based on the processed image data
        self.publish_control(steering_angle, throttle)

    def find_largest_white_group_center(self, row_pixels):
        """
        Given a row (an array of pixels), find the largest consecutive group
        of white pixels (where a white pixel is [255,255,255]). Return the center
        column index of that group and the (start, end) tuple of that group.
        If no white pixel is found, returns (None, None).
        """
        best_start = None
        best_end = None
        best_length = 0

        current_start = None
        current_length = 0

        for col, pixel in enumerate(row_pixels):
            if (pixel == [255, 255, 255]).all():
                if current_start is None:
                    current_start = col
                    current_length = 1
                else:
                    current_length += 1
            else:
                if current_start is not None:
                    if current_length > best_length:
                        best_length = current_length
                        best_start = current_start
                        best_end = col - 1
                    current_start = None
                    current_length = 0

        # Check if the row ended while we were in a white sequence.
        if current_start is not None and current_length > best_length:
            best_length = current_length
            best_start = current_start
            best_end = len(row_pixels) - 1

        if best_start is not None:
            center = (best_start + best_end) // 2
            return center, (best_start, best_end)
        else:
            return None, None

    def find_lane_lines(self, img):
        """
        Revised lane detection that:
        - Scans from mask_height downward until a row is found that contains white.
            In that row the white pixel chosen is the center of the largest consecutive
            white segment. This gives start_pt (start_pt_col at row start_pt_row).
        - If start_pt is within 50 pixels of the left or right edge, then that lane is lost.
        - For each row from start_pt_row to the bottom, starting from the start_pt, it
            scans right and left for the first black pixel transition.
        
        Returns:
        left_lane_edges:  Dictionary mapping row indices to left lane edge column positions (or None)
        right_lane_edges: Dictionary mapping row indices to right lane edge column positions (or None)
        start_pt_row:     The row at which the first white region was found.
        """
        height, width, _ = img.shape
        start_pt_row = None
        start_pt_col = None

        # 1. Find the first row (starting from mask_height) that contains white.
        for row in range(self.mask_height, height):
            row_pixels = img[row]
            center, group = self.find_largest_white_group_center(row_pixels)
            if center is not None:
                start_pt_row = row
                start_pt_col = center
                break

        if start_pt_row is None:
            # No white region was found anywhere.
            return {}, {}, None, None

        # 2. Determine if a lane is missing from the start row.
        no_left_lane = height - start_pt_row < 120
        # print(f"Height: {height - start_pt_row}")
        no_right_lane = False
        if no_left_lane:
            print("No left lane detected.")
        if no_right_lane:
            print("No right lane detected.")

        left_lane_edges = {}
        right_lane_edges = {}

        # 3. Process every row from start_pt_row to the bottom of the image.
        for row in range(start_pt_row, height):
            row_pixels = img[row]

            # 4a. Determine the right lane edge.
            if not no_right_lane:
                right_edge = None
                # Starting from start_pt_col, move right until a non-white pixel is found.
                for col in range(start_pt_col, width):
                    if not (row_pixels[col] == [255, 255, 255]).all():
                        right_edge = col
                        break
                if right_edge is None:
                    right_edge = width - 1
                right_lane_edges[row] = right_edge

            # 4b. Determine the left lane edge.
            if not no_left_lane:
                left_edge = None
                # Starting from start_pt_col, move left until a non-white pixel is found.
                for col in range(start_pt_col, -1, -1):
                    if not (row_pixels[col] == [255, 255, 255]).all():
                        left_edge = col
                        break
                if left_edge is None:
                    left_edge = 0
                left_lane_edges[row] = left_edge

        return right_lane_edges, left_lane_edges, start_pt_row, start_pt_col

    def draw_edge_line(self, edge_pts, img):
        """
        Fits a 2nd-degree polynomial to the provided edge points and draws the fitted
        line in bright yellow on the image.
        """
        if len(edge_pts) < 10:
            return None
        
        # Get row indices and corresponding edge column positions.
        x_vals = list(edge_pts.keys())
        y_vals = [edge_pts[x] for x in x_vals]
        
        # Fit a 2nd-degree polynomial.
        z = np.polyfit(x_vals, y_vals, 3)
        f = np.poly1d(z)
        
        # Draw the fitted line in bright yellow (BGR: 0,255,255).
        for r in range(min(x_vals), max(x_vals)):
            c = int(f(r))
            if 0 <= c < img.shape[1]:
                img[r, c] = (0, 255, 255)
        return f

    def find_mid_points(self, img, right_pts, left_pts, right_fit, left_fit, start_pt_row):
        """
        Combines left and right lane edge points to find the overall mid-lane for each row.
        """
        mid_pts = {}
        max_h = img.shape[0]
        if right_fit is None and left_fit is None:
            print("No fits")
            return mid_pts
        for row in range(start_pt_row, max_h):
            r_col = img.shape[1] - 1
            l_col = 0
            if (row not in right_pts) and (row not in left_pts):
                continue  # Skip rows where no edge data is available.
            if left_fit is None:
                r_col = right_pts[row] if row in right_pts else right_fit(row)
            elif right_fit is None:
                l_col = left_pts[row] if row in left_pts else left_fit(row)
            else:
                # Use available data; if missing, compute using the fitted polynomial.
                r_col = right_pts[row] if row in right_pts else right_fit(row)
                l_col = left_pts[row] if row in left_pts else left_fit(row)

            mid_col = int(0.5 * (r_col + l_col))
            if 0 <= mid_col < img.shape[1]:
                mid_pts[row] = mid_col
        return mid_pts

    def find_ideal_path(self, img, mid_pts):
        """Fits a 3rd-degree polynomial to the mid-lane points and draws a red line along it.
        
        Args:
            mid_pts (dict): A dictionary mapping row indices to mid-lane column positions.
            img (numpy.ndarray): The image on which to draw the polynomial.
            
        Returns:
            np.poly1d: The fitted polynomial function, or None if insufficient points.
        """
        if len(mid_pts) < 10:
            return None

        # Extract x (row indices) and y (column positions) from mid_pts.
        x_vals = list(mid_pts.keys())
        y_vals = [mid_pts[x] for x in x_vals]
        
        # Fit a 3rd-degree polynomial.
        z = np.polyfit(x_vals, y_vals, 1)
        f = np.poly1d(z)
        
        # Draw the polynomial: for each row in the image, compute the column from f(row)
        for r in range(self.mask_height, img.shape[0]):
            c = int(f(r))
            if 0 <= c < img.shape[1]:
                img[r, c] = (255, 0, 0)  # Blue in BGR format.
                
        return f

    def find_center_offset(self, img, mid_lane_fit, lookahead_px):
        """How far from center at lookahead row."""
        if mid_lane_fit is None:
            return 0.0
        col = mid_lane_fit(lookahead_px)
        center = img.shape[1] / 2.0
        print(f"Center: {center:.3f}, Col: {col:.3f}")
        return center - col  # Positive if mid-lane is right of center, negative if left.

    def process_image(self, image: np.ndarray):
        # Apply mask to focus on the road
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, self.mask_vertices, (255, 255, 255))
        masked = cv2.bitwise_and(image, mask)

        # # 1) Convert masked image to grayscale
        # gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # # 2) Resize the grayscale image
        # small_gray = cv2.resize(
        #     gray_masked,
        #     (int(gray_masked.shape[1] * self.scale_down_width_factor),
        #     int(gray_masked.shape[0] * self.scale_down_height_factor))
        # )

        # # 3) Now blur the single-channel image
        # blurred = cv2.GaussianBlur(small_gray, (5, 5), 0)

        # # 4) Threshold the single-channel blurred image
        # _, binary = cv2.threshold(blurred, self.thresh_value, 255, cv2.THRESH_BINARY)

        # # 5) Erode / morph
        # thresh = cv2.erode(binary, self.erode_kernel, iterations=1)

        # # 6) Convert back to RGB
        # binary_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        #1 Convert masked image to hsv
        hsv_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        # 2 Get V channel, which represents brightness
        v_chan = hsv_masked[:, :, 2]

        # 3 Resize
        small_v = cv2.resize(
            v_chan,
            (int(v_chan.shape[1] * self.scale_down_width_factor),
            int(v_chan.shape[0] * self.scale_down_height_factor))
        )

        # 4 Blur
        v_blurred = cv2.GaussianBlur(small_v, (11, 11), 0)

        # 5 Threshold
        _, binary = cv2.threshold(v_blurred, self.thresh_value, 255, cv2.THRESH_BINARY)

        # 6 Erode
        eroded = cv2.erode(binary, self.erode_kernel, iterations=1)

        # Convert back to RGB, we use GRAY2RGB here because eroded is a single channel representing brightness
        binary_color = cv2.cvtColor(eroded, cv2.COLOR_GRAY2RGB)

        lookahead_px = int(binary_color.shape[0]*(1 - self.lookahead_distance))

        if 0 <= lookahead_px < binary_color.shape[0]:
            binary_color[lookahead_px, :] = (127,127,127)

        right_edges, left_edges, start_pt_row, start_pt_col = self.find_lane_lines(binary_color)

        right_fit = self.draw_edge_line(right_edges, binary_color)
        left_fit = self.draw_edge_line(left_edges, binary_color)

        # Compute midpoints and draw the mid-lane in red.
        mid_points = self.find_mid_points(binary_color, right_edges, left_edges, right_fit, left_fit, start_pt_row)
        
        # # Update prev_mid_pts (if desired) for smoother tracking.
        # self.prev_mid_pts = mid_points

        # Find the ideal path (3rd-degree polynomial on mid-lane).
        ideal_path = self.find_ideal_path(binary_color, mid_points)
        error = self.find_center_offset(binary_color, ideal_path, lookahead_px)

        cv2.line(
            binary_color,
            (binary_color.shape[1] // 2, 0),                  # starting point: center top
            (binary_color.shape[1] // 2, binary_color.shape[0]),# ending point: center bottom
            (0, 0, 255),                                      # color in red here
            1                                                 # thickness
        )
        cv2.line(
            binary_color,
            (start_pt_col, 0),                  # starting point: center top
            (start_pt_col, binary_color.shape[0]),# ending point: center bottom
            (0, 255, 255),                                      # color in red here
            1                                                 # thickness
        )
        cv2.imshow("Processed Image", binary_color)

        current_time = time.time()
        dt = current_time - self.last_time if current_time - self.last_time > 0 else 0.01
        self.last_time = current_time

        # Update the integral and derivative components
        self.pid_integral += error * dt
        derivative = (error - self.pid_prev_error) / dt
        self.pid_prev_error = error

        # Compute PID output for steering
        steering_angle = self.pid_kp * error + self.pid_ki * self.pid_integral + self.pid_kd * derivative
        
        max_steering = np.deg2rad(90)  # ±90° limit
        steering_angle = np.clip(steering_angle, -max_steering, max_steering)

        # reduce throttle (speed) when the error is large since its usually a big turn
        max_throttle = 0.5
        throttle = max_throttle * (1 - min(abs(error) / (binary_color.shape[1] / 2), 1))
        # throttle = max_throttle
        print(f"Error: {error:.3f}, Steering Angle: {np.rad2deg(steering_angle):.3f}, Throttle: {throttle:.3f}")

        cv2.waitKey(1)

        return steering_angle, throttle

    def publish_control(self, steering, throttle):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = throttle
        self.drive_pub.publish(drive_msg)
        self.get_logger().info(f'Published steering: {steering:.3f}, throttle: {throttle:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
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