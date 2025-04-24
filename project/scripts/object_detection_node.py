#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import time

# YOLO library
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Topics
        self.rgb_topic = '/color/image_raw'
        self.depth_topic = '/depth/image_rect_raw'
        self.enable_detection_topic = '/enable_detection'
        self.detected_sign_topic = '/detected_sign'
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, 10
        )
        # Subscribe to “enable_detection” so we know whether to pause
        self.enable_detection_sub = self.create_subscription(
            Bool, self.enable_detection_topic, self.enable_detection_callback, 10
        )

        # Publisher
        self.sign_pub = self.create_publisher(String, self.detected_sign_topic, 10)

        # Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Detection state
        self.enable_detection = True
        
        # Store latest images so we can process them once both arrive
        self.latest_rgb = None
        self.latest_depth = None

        # YOLO model
        self.model_path = "../yolo11s_model/model.pt"  # YOLO model path
        self.confidence_threshold = 0.7
        self.model = YOLO(self.model_path)
        self.model.to('cuda')  # Use GPU if available

        # Class names (labels)
        self.labels = self.model.names
        self.bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                            (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]
        
        # Display Image
        self.display_image = True
        self.output_folder = "/home/jetson/f1tenth_ws/src/team-a3-redbull/milestone5/reference_image"

        # Debugging
        self.print_object_detection = True

        print("ObjectDetectionNode initialized.")

    def enable_detection_callback(self, msg: Bool):
        """Pause or resume detections based on this Boolean topic."""
        self.enable_detection = msg.data
        if not self.enable_detection:
            print("Detection is now DISABLED (e.g., during U-turn).")
        else:
            print("Detection is now ENABLED.")

    def rgb_callback(self, msg: Image):
        """Callback for the RGB camera."""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'RGB CvBridge Error: {e}')
            return
        
        if rgb_image.dtype != np.float32:
            rgb_image = rgb_image.astype(np.float32)

        self.latest_rgb = rgb_image
        self.process_and_annotate_images()

    def depth_callback(self, msg: Image):
        """Callback for the Depth camera."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Depth CvBridge Error: {e}')
            return

        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)

        self.latest_depth = depth_image
        self.process_and_annotate_images()
        
    def process_and_annotate_images(self):
        if (self.latest_rgb is not None) and (self.latest_depth is not None) and self.enable_detection:
            rgb_img = self.latest_rgb.copy()
            depth_img = self.latest_depth.copy()
            depth_img = self.zoom_in_image(depth_img, 1.8) # Zoom in on depth image

            results = self.model(rgb_img, verbose=False)
            detections = results[0].boxes
            if len(detections) > 0:
              self.annotate_and_display(rgb_img, depth_img, detections)

            self.latest_rgb = None  # Reset to avoid reprocessing
            self.latest_depth = None  # Reset to avoid reprocessing

    def zoom_in_image(self, image, zoom_factor):
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Compute crop size
        new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
        x1 = max(center_x - new_w // 2, 0)
        y1 = max(center_y - new_h // 2, 0)
        x2 = min(center_x + new_w // 2, w)
        y2 = min(center_y + new_h // 2, h)

        cropped = image[y1:y2, x1:x2]

        # Resize back to original shape
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def annotate_and_display(self, rgb_img, depth_img, detections):
        object_count = 0
        detection_results = []  # For printing
        found = False
        depth_colormap = self.colorize_depth(depth_img)

        for detection in detections:
            conf = detection.conf.item()
            classidx = int(detection.cls.item())
            classname = self.labels[classidx]

            if conf > self.confidence_threshold:
                xyxy = detection.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy)

                # Adjust the bounding box for depth image
                shift_x = 0              # Pixels to shift left
                scale_factor = 0.6        # Scale down the box to 60% of original

                # Original box center
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                box_width = x2 - x1
                box_height = y2 - y1

                # Scale down
                new_width = int(box_width * scale_factor)
                new_height = int(box_height * scale_factor)

                # Shift and compute new corners
                x1_depth = max(0, box_center_x - new_width // 2 - shift_x)
                x2_depth = max(0, box_center_x + new_width // 2 - shift_x)
                y1_depth = max(0, box_center_y - new_height // 2)
                y2_depth = max(0, box_center_y + new_height // 2)

                avg_depth = self.get_average_depth(depth_img, (x1_depth, y1_depth, x2_depth, y2_depth))

                # Update for printing
                detection_results.append({
                    "class": classname,
                    "confidence": conf,
                    "avg_depth": avg_depth,
                    "bbox": (x1, y1, x2, y2)
                })
                object_count += 1
                
                # Publish the detected sign with average depth
                message_str = f"{classname}|{avg_depth:.2f}"
                msg = String()
                msg.data = message_str
                self.sign_pub.publish(msg)
                
                # Draw on the images if display is enabled
                if self.display_image:
                    found = True

                    color = self.bbox_colors[classidx % len(self.bbox_colors)]
                    label = f"{classname}: {int(conf * 100)}%"

                    # Draw on RGB
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color, 2)
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(y1, label_size[1] + 10)
                    cv2.rectangle(rgb_img, (x1, label_ymin - label_size[1] - 10), 
                                  (x1 + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
                    cv2.putText(rgb_img, label, (x1, label_ymin - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # Draw on Depth
                    cv2.rectangle(depth_colormap, (x1_depth, y1_depth), (x2_depth, y2_depth), color, 2)
                    cv2.putText(depth_colormap, f"Avg Depth: {avg_depth:.2f}", (x1_depth, y2_depth + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if found and self.display_image:
            cv2.putText(rgb_img, f"Objects detected: {object_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            depth_resized = cv2.resize(depth_colormap, (rgb_img.shape[1], rgb_img.shape[0]))
            combined = np.hstack((rgb_img, depth_resized))
            os.makedirs(self.output_folder, exist_ok=True)
            filename = os.path.join(self.output_folder, f"object_detect_image_{int(time.time()*1000)}.png") # Save with unique timestamp
            # filename = os.path.join(self.output_folder, "object_detect_image.png")
            cv2.imwrite(filename, combined)
            cv2.waitKey(1)
        
        if self.print_object_detection:
            print("Objects detected:", object_count)
            for res in detection_results:
                print(f"{res['class']} - {int(res['confidence'] * 100)}% - Avg Depth: {res['avg_depth']:.2f} m - BBox: {res['bbox']}")

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

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
