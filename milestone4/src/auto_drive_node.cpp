#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

#include <deque>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>

#define DEG2RAD(x) ((x)*M_PI/180.0)
#define RAD2DEG(x) ((x)*180.0/M_PI)
#define SIGN(x) ((x) > 0 ? 1 : -1)

class AutoDrive : public rclcpp::Node {
public:
    AutoDrive() :
        // Field initializations
        Node("auto_drive_node"), 
        speed(0.0),
        bubble_radius(0.45),
        range_angle(DEG2RAD(110)),
        min_gap_distance(0.5),
        kp(0.65),
        kd(0.15),
        ki(0.0),
        pid_integral(0.0),
        pid_prev_error(0.0),
        dt(0.05),
        is_truncated(false),
        stop_threshold(2.0),
        stop(false),
        large_angle_threshold(DEG2RAD(10)),
        lap_count(0),
        detect_threshold(0.9),
        ignoring(false),
        ignoring_duration(5.0),
        reference_set(false)
    {
        // Declare and get parameters
        this->declare_parameter<int>("num_laps", -1);
        this->get_parameter("num_laps", num_laps);
        this->declare_parameter<bool>("USING_SIM", true);
        this->get_parameter("USING_SIM", USING_SIM);

        // Create publishers, subscribers, and timers
        publisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        scan_subscription = this->create_subscription<sensor_msgs::msg::LaserScan>("/scan", 10, std::bind(&AutoDrive::scan_callback, this, std::placeholders::_1));
        image_subscription = this->create_subscription<sensor_msgs::msg::Image>("/camera/camera/color/image_raw", 10, std::bind(&AutoDrive::image_callback, this, std::placeholders::_1));
        timer = this->create_wall_timer(std::chrono::seconds(3), std::bind(&AutoDrive::timer_callback, this));


        RCLCPP_INFO(this->get_logger(), "AutoDrive C++ Initialized");
    }

private:
// ================================================== VARIABLES ==================================================


    // ROS2 publishers and subscribers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher; // Publisher for drive commands
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription; // Subscriber for LiDAR scans
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription; // Subscriber for camera images

    // Driving parameters
    double speed; // Speed of the car (m/s)
    double bubble_radius; // Safety bubble radius (m)
    double range_angle; // Angle of the lidar range (rad)
    double min_gap_distance; // Minimum gap distance (m)
    double kp; // Proportional gain
    double kd; // Derivative gain
    double ki; // Integral gain
    double pid_integral; // Error integral term
    double pid_prev_error; // Previous error term
    double dt; // Time step (s)
    bool is_truncated; // Flag to check if the LiDAR scan has been truncated
    int truncated_start_index; // Start index of the truncated LiDAR scan
    int truncated_end_index; // End index of the truncated LiDAR scan
    double stop_threshold; // Threshold for stopping the car (m)
    bool stop; // Flag to stop the car
    double large_angle_threshold; // Threshold for large steering angles (rad)
    
    // Lap counting
    int num_laps; // Number of laps to complete
    int lap_count; // Number of laps completed
    double detect_threshold; // Threshold for detecting the reference image
    bool ignoring; // Flag to ignore lap counting
    double ignoring_start_time; // Time to start ignoring lap counting
    double ignoring_duration; // Duration to ignore lap counting
    bool reference_set; // Flag to check if the reference image has been set
    cv::Mat processed_ref; // Processed reference image
    std::string output_folder = "/home/jetson/f1tenth_ws/src/team-a3-redbull/milestone4/reference_image";

    // History buffers
    std::deque<double> largest_gap_distance_history; // History of the largest gap distances (m)
    std::deque<int> largest_gap_history; // History of the largest gap sizes (number of points)
    std::deque<double> speed_history; // History of the car speeds (m/s)
    const size_t largest_gap_distance_history_size = 20;
    const size_t largest_gap_history_size = 5;
    const size_t speed_history_size = 10;

    // Debugging stuff
    rclcpp::TimerBase::SharedPtr timer;
    bool USING_SIM;


// ================================================== CALLBACK FUNCTIONS ==================================================


    /**
     * Callback function for the camera image messages.
     * This function is used to count laps based on the reference image.
     * 
     * @param msg Camera image message.
     * @return void.
     */
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            cv::Mat image = cv_ptr->image;
            cv::Mat processed = process_image_for_comparison(image);

            // Set the reference image if not already set
            if (!reference_set) {
                processed_ref = processed.clone();
                reference_set = true;
                cv::imwrite(output_folder + "/reference_image_processed.png", processed_ref);
                RCLCPP_INFO(this->get_logger(), "Reference image saved");
                return;
            }

            // Ignore lap counting if already counting
            if (ignoring) {
                double elapsed = this->now().seconds() - ignoring_start_time;
                if (elapsed > ignoring_duration) {
                    ignoring = false;
                }
                return;
            }

            // Compare the processed image with the reference image
            cv::Mat result;
            cv::matchTemplate(processed, processed_ref, result, cv::TM_CCOEFF_NORMED);
            double max_val;
            cv::minMaxLoc(result, nullptr, &max_val, nullptr, nullptr);

            // Count a lap if the match is above the detection threshold
            if (max_val >= detect_threshold) {
                lap_count++;
                RCLCPP_INFO(this->get_logger(), "Lap counted! Total: %d", lap_count);
                ignoring = true;
                ignoring_start_time = this->now().seconds();
            }
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge error: %s", e.what());
        }
    }


    /**
     * Callback function for the LiDAR scan messages.
     * This function implements the main logic for the autonomous driving.
     * 
     * @param scan_msg LiDAR scan message.
     * @return void.
     */
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) {
        // If no reference image is set, do not proceed
        if (!USING_SIM && !reference_set) {
            RCLCPP_WARN(this->get_logger(), "Reference image not set: Shutting down");
            rclcpp::shutdown();
            std::exit(1);
        }

        // Stay stopped if safety stop is triggered, or if the lap limit is reached
        if (stop || check_laps()) {
            if (check_laps())
                RCLCPP_INFO(this->get_logger(), "Laps completed: Shutting down");
            rclcpp::shutdown();
            std::exit(0);
        }

        // Preprocess the LiDAR scan
        auto filtered_ranges = preprocess_lidar_scan(scan_msg);

        // Find the closest point and draw a safety bubble around it
        int closest_index = std::min_element(filtered_ranges.begin(), filtered_ranges.end()) - filtered_ranges.begin();
        draw_safety_bubble(filtered_ranges, closest_index);

        // Find the best gap
        auto [start_index, end_index] = find_max_gap(filtered_ranges);

        // Check if the car should stop
        if (new_safety_stop(scan_msg, filtered_ranges, start_index, end_index)) return;

        // Find the best point in the gap and calculate the steering angle
        int best_point_index = find_best_point(start_index, end_index);
        int midpoint = filtered_ranges.size() / 2;
        double error_angle = (best_point_index - midpoint) * scan_msg->angle_increment;

        // Calculate the steering angle and speed, and publish the control message
        double steering_angle = pid_control(error_angle);
        double speed = get_speed(steering_angle, filtered_ranges, best_point_index);
        publish_control(steering_angle, speed);
    }


    /**
     * Timer callback function for debugging purposes.
     * 
     * @param timer Timer object.
     * @return void.
     */
    void timer_callback() {
        if (stop) return;
        RCLCPP_INFO(this->get_logger(), "Timer callback");
        RCLCPP_INFO(this->get_logger(), "Lap count: %d / %d", lap_count, num_laps);
        RCLCPP_INFO(this->get_logger(), "Speed: %.2f m/s", speed);
        std::cout << std::endl;
    }


// ================================================== CONTROL AND DECISION FUNCTIONS ==================================================


    /**
     * Finds the best point in the gap to steer towards (the center of the gap).
     * 
     * @param start_index Start index of the gap.
     * @param end_index End index of the gap.
     * @return Index of the best point in the gap.
     */
    int find_best_point(int start_index, int end_index) {
        return (start_index + end_index) / 2;
    }


    /**
     * Checks if the car has reached the lap limit.
     * 
     * @return True if the lap limit has been reached, false otherwise.
     */
    bool check_laps() {
        if (num_laps != -1 && lap_count > num_laps) {
            publish_control(0.0, 0.0);
            RCLCPP_INFO(this->get_logger(), "Lap limit reached: Now stopping");
            stop = true;
            return true;
        }
        return false;
    }


    /**
     * Get the speed of the car based on the steering angle and the LiDAR ranges.
     * 
     * @param angle Steering angle of the car (rad).
     * @param ranges Filtered LiDAR ranges.
     * @param best_point_index Index of the best point in the gap.
     * @return Speed of the car (m/s).
     */
    double get_speed(double angle, const std::vector<double> &ranges, int best_point_index) {
        double abs_deg = std::abs(RAD2DEG(angle));
        double speed;

        if (abs_deg <= 10) {
            speed = 1.2 + std::exp(0.04 * ranges[best_point_index]);
        } else if (abs_deg <= 15) {
            speed = 2.5;
        } else {
            speed = 1.5;
        }
        speed = std::min(speed, 3.0);

        return speed;
    }


    /**
     * Finds the largest gap in the LiDAR ranges.
     * 
     * @param ranges Filtered LiDAR ranges.
     * @return Start and end indices of the largest gap.
     */
    std::pair<int, int> find_max_gap(const std::vector<double> &ranges) {
        int max_start = 0, max_size = 0;
        int current_start = 0, current_size = 0;

        for (size_t i = 0; i < ranges.size(); i++) {
            if (ranges[i] > min_gap_distance) {
                if (current_size == 0)
                    current_start = i;
                current_size++;
            } else {
                if (current_size > max_size) {
                    max_start = current_start;
                    max_size = current_size;
                }
                current_size = 0;
            }
        }

        if (current_size > max_size) {
            max_start = current_start;
            max_size = current_size;
        }

        return {max_start, max_start + max_size};
    }


    /**
     * PID controller for the steering angle.
     * 
     * @param error Error (difference between the desired and actual steering angle).
     * @return Steering angle.
     */
    double pid_control(double error) {
        // Update PID terms
        pid_integral += error * dt;
        double derivative = (error - pid_prev_error) / dt;
        pid_prev_error = error;

        // Calculate the steering angle
        double angle = kp * error + ki * pid_integral + kd * derivative;
        angle = std::clamp(angle, -DEG2RAD(90.0), DEG2RAD(90.0));

        // Apply additional steering angle adjustments
        if (std::abs(angle) < DEG2RAD(4.5)) {
            angle = std::pow(angle, 2) * SIGN(angle) * 0.7;
        } else if (std::abs(angle) < DEG2RAD(7)) {
            angle = std::pow(angle, 2) * SIGN(angle) * 1.0;
        } else {
            angle *= 0.75;
        }

        // Apply a large angle threshold
        if (std::abs(angle) > large_angle_threshold) {
            angle *= 1.8;
        }

        return angle;
    }


    /**
     * Publishes an AckermannDriveStamped message to the car.
     * 
     * @param angle Steering angle of the car (rad).
     * @param speed Speed of the car (m/s).
     * @return void.
     */
    void publish_control(double angle, double speed) {
        // Send the drive message to the car
        auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
        drive_msg.drive.steering_angle = angle;
        drive_msg.drive.speed = speed;
        publisher->publish(drive_msg);

        // Update the speed history
        this->speed = speed;
        speed_history.push_back(speed);
        if (speed_history.size() > speed_history_size) {
            speed_history.pop_front();
        }

        // Update the stop threshold based on the average speed
        double sum_speed = std::accumulate(speed_history.begin(), speed_history.end(), 0.0);
        double avg_speed = sum_speed / speed_history.size();
        if (avg_speed <= 1.5) {
            stop_threshold = 1.5;
        } else if (avg_speed <= 2.0) {
            stop_threshold = 2.5;
        } else if (avg_speed <= 3.0) {
            stop_threshold = 3.0;
        } else if (avg_speed <= 4.0) {
            stop_threshold = 3.5;
        } else {
            stop_threshold = 4.0;
        }
    }



// ================================================== PROCESSING FUNCTIONS ==================================================


    /**
     * Converts the input image to grayscale, masks out certain areas of the image,
     * applies binary thresholding, and fills in any white spots within black regions.
     * 
     * @param image Input image.
     * @return Processed image.
     */
    cv::Mat process_image_for_comparison(const cv::Mat& image) {
        cv::Mat gray, masked, bw, inv_bw, inv_bw_closed;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Mask dimensions
        int h = gray.rows;
        int w = gray.cols;
        int mask_height = static_cast<int>(0.3 * h); // Keep top 30% of the image
        int mask_left = static_cast<int>(0.05 * w); // Mask out 5% of the image on the left
        int mask_right = static_cast<int>(0.70 * w); // Mask out 30% of the image on the right (keeping left 70%)
        
        // Apply mask
        masked = cv::Mat::zeros(gray.size(), gray.type());
        gray(cv::Rect(mask_left, 0, mask_right - mask_left, mask_height)).copyTo(masked(cv::Rect(mask_left, 0, mask_right - mask_left, mask_height)));

        // Apply binary thresholding and fill white spots
        cv::threshold(masked, bw, 127, 255, cv::THRESH_BINARY);
        cv::bitwise_not(bw, inv_bw);
        cv::morphologyEx(inv_bw, inv_bw_closed, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
        cv::bitwise_not(inv_bw_closed, bw);

        return bw;
    }


    /**
     * Preprocesses the LiDAR scan message by truncating the range and applying a smoothing filter.
     * 
     * @param scan_msg LiDAR scan message.
     * @return Filtered LiDAR ranges.
     */
    std::vector<double> preprocess_lidar_scan(const sensor_msgs::msg::LaserScan::SharedPtr &scan_msg) {
        // Truncate the range of the LiDAR scan
        if (!is_truncated) {
            int total_range = scan_msg->ranges.size();
            int range_size = static_cast<int>((range_angle / (scan_msg->angle_max - scan_msg->angle_min)) * total_range);
            truncated_start_index = (total_range / 2) - (range_size / 2);
            truncated_end_index = (total_range / 2) + (range_size / 2);
            is_truncated = true;
        }

        // Need to "shift" the indexes to start from zero
        std::vector<double> filtered_ranges;
        for (int i = truncated_start_index; i < truncated_end_index; i++) {
            if (std::isnan(scan_msg->ranges[i])) {
                filtered_ranges.push_back(0.0); // set to zero if NaN
            } else if (scan_msg->ranges[i] > scan_msg->range_max || std::isinf(scan_msg->ranges[i])) {
                filtered_ranges.push_back(scan_msg->range_max);
            } else {
                filtered_ranges.push_back(scan_msg->ranges[i]);
            }
        }

        // Apply a smoothing filter to the LiDAR ranges
        std::vector<double> smooth_ranges;
        int window = 5; // Note: actual window size is 2*window + 1
        for (size_t i = window; i < filtered_ranges.size() - window; i++) {
            double min_val = std::numeric_limits<double>::max();
            for (int j = -window; j <= window; j++) {
                if (filtered_ranges[i + j] < min_val) {
                    min_val = filtered_ranges[i + j];
                }
            }
            smooth_ranges.push_back(min_val);
        }

        return smooth_ranges;
    }


    /**
     * Draws a safety bubble (zeros the points) around the closest point in the LiDAR ranges.
     * 
     * @param ranges Filtered LiDAR ranges.
     * @param center_index Index of the closest point.
     * @return void.
     */
    void draw_safety_bubble(std::vector<double> &ranges, int center_index) {
        double center_distance = ranges[center_index];
        ranges[center_index] = 0.0;

        // Zero out points to the right (in the array)
        for (size_t i = center_index + 1; i < ranges.size(); i++) {
            if (ranges[i] > center_distance + bubble_radius) break;
            ranges[i] = 0.0;
        }

        // Zero out points to the left (in the array)
        for (int i = center_index - 1; i >= 0; i--) {
            if (ranges[i] > center_distance + bubble_radius) break;
            ranges[i] = 0.0;
        }
    }


    /**
     * Checks if the car should stop based on the gap distance and width.
     * 
     * @param scan_msg LiDAR scan message.
     * @param ranges Filtered LiDAR ranges.
     * @param start_index Start index of the gap.
     * @param end_index End index of the gap.
     * @return True if the car should stop, false otherwise.
     */
    bool new_safety_stop(const sensor_msgs::msg::LaserScan::SharedPtr &scan_msg, const std::vector<double> &ranges, int start_index, int end_index) {
        double gap_max = *std::max_element(ranges.begin() + start_index, ranges.begin() + end_index);
        largest_gap_distance_history.push_back(gap_max);
        if (largest_gap_distance_history.size() > largest_gap_distance_history_size) {
            largest_gap_distance_history.pop_front();
        }

        if (std::accumulate(largest_gap_distance_history.begin(), largest_gap_distance_history.end(), 0.0) / largest_gap_distance_history.size() < stop_threshold) {
            publish_control(0.0, 0.0);
            RCLCPP_WARN(this->get_logger(), "EMERGENCY STOP: Gap distance");
            stop = true;
            return true;
        }

        int gap_size = end_index - start_index;
        largest_gap_history.push_back(gap_size);
        if (largest_gap_history.size() > largest_gap_history_size) {
            largest_gap_history.pop_front();
        }

        double gap_threshold = DEG2RAD(30) / scan_msg->angle_increment;
        if (std::accumulate(largest_gap_history.begin(), largest_gap_history.end(), 0.0) / largest_gap_history.size() < gap_threshold) {
            publish_control(0.0, 0.0);
            RCLCPP_WARN(this->get_logger(), "EMERGENCY STOP: Gap width");
            stop = true;
            return true;
        }

        return false;
    }
};


// ================================================== MAIN FUNCTION ==================================================


int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutoDrive>());
    rclcpp::shutdown();
    return 0;
}
