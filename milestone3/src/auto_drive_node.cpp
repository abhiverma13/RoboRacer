#include "rclcpp/rclcpp.hpp"
#include <string>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <cmath>

#include <memory>
#include <vector>
#include <iostream>

#define DEG2RAD(x) ((x)*M_PI/180.0)
#define RAD2DEG(x) ((x)*180.0/M_PI)

class
AutoDrive : public rclcpp::Node {

public:
    AutoDrive() : Node("auto_drive_node"), speed_(0.0), bubble_radius_(0.35), range_angle_(M_PI * 110 / 180),
                  kp_(0.65), kd_(0.15), ki_(0.0), pid_integral_(0.0), pid_prev_error_(0.0), dt_(0.05),
                  is_truncated_(false), truncated_start_index_(0), truncated_end_index_(0) {

        // Create publisher and subscriber
        publisher_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);
        scan_subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&AutoDrive::scan_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "AutoDrive Initialized");
    }

private:
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;

    double speed_;
    double bubble_radius_;
    double range_angle_;
    double kp_, kd_, ki_;
    double pid_integral_, pid_prev_error_;
    double dt_;

    bool is_truncated_;
    int truncated_start_index_, truncated_end_index_;

    double get_speed(double angle, const std::vector<double> &ranges, int best_point_index) {
        double abs_angle = std::fabs(angle * 180.0 / M_PI);
        if (abs_angle <= 10) {
            return 0.2 + std::exp(0.04 * std::fabs(ranges[best_point_index]));
        } 
        else if (abs_angle <= 20) return 1.0;
        else if (abs_angle <= 30) return 0.7;
        else return 0.5;
    }

    std::vector<double> preprocess_lidar_scan(const sensor_msgs::msg::LaserScan::SharedPtr &scan_msg) {
        if (!is_truncated_) {
            int total_range = scan_msg->ranges.size();
            int range_size = static_cast<int>((range_angle_ / (scan_msg->angle_max - scan_msg->angle_min)) * total_range);
            truncated_start_index_ = (total_range / 2) - (range_size / 2);
            truncated_end_index_ = (total_range / 2) + (range_size / 2);
            is_truncated_ = true;
        }

        std::vector<double> filtered_ranges;
        for (int i = truncated_start_index_; i < truncated_end_index_; ++i) {
            if (std::isnan(scan_msg->ranges[i]) || scan_msg->ranges[i] > scan_msg->range_max || std::isinf(scan_msg->ranges[i])) {
                filtered_ranges.push_back(scan_msg->range_max);
            } else {
                filtered_ranges.push_back(scan_msg->ranges[i]);
            }
        }

        std::vector<double> smooth_ranges;
        int window = 5;
        for (size_t i = window; i < filtered_ranges.size() - window; ++i) {
            double sum = 0.0;
            for (int j = -window; j <= window; ++j) {
                sum += filtered_ranges[i + j];
            }
            smooth_ranges.push_back(sum / (2 * window + 1));
        }
        return smooth_ranges;
    }

    void draw_safety_bubble(std::vector<double> &ranges, int center_index) {
        double center_distance = ranges[center_index];
        ranges[center_index] = 0.0;

        for (size_t i = center_index + 1; i < ranges.size(); ++i) {
            if (ranges[i] > center_distance + bubble_radius_) break;
            ranges[i] = 0.0;
        }

        for (int i = center_index - 1; i >= 0; --i) {
            if (ranges[i] > center_distance + bubble_radius_) break;
            ranges[i] = 0.0;
        }
    }

    std::pair<int, int> find_max_gap(const std::vector<double> &ranges) {
        int max_start = 0, max_size = 0;
        int current_start = 0, current_size = 0;

        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i] > 0.1) {
                if (current_size == 0) {
                    current_start = i;
                }
                ++current_size;
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

    int find_best_point(int start_index, int end_index) {
        return (start_index + end_index) / 2;
    }

    double pid_control(double error) {
        pid_integral_ += error * dt_;
        double derivative = (error - pid_prev_error_) / dt_;
        pid_prev_error_ = error;

        double angle = kp_ * error + ki_ * pid_integral_ + kd_ * derivative;
        double max_angle = DEG2RAD(90);
        if(angle < -max_angle) {
            return -max_angle;
        } else if(angle > max_angle) {
            return max_angle;
        }
        return angle;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr scan_msg) {
        auto filtered_ranges = preprocess_lidar_scan(scan_msg);

        int closest_index = std::min_element(filtered_ranges.begin(), filtered_ranges.end()) - filtered_ranges.begin();
        draw_safety_bubble(filtered_ranges, closest_index);

        auto gap_indices = find_max_gap(filtered_ranges);
        int start_index = gap_indices.first;
        int end_index = gap_indices.second;
        int best_point_index = find_best_point(start_index, end_index);

        int midpoint = filtered_ranges.size() / 2;
        double error_index = best_point_index - midpoint;
        double error_angle = error_index * scan_msg->angle_increment;

        double steering_angle = pid_control(error_angle);

        ackermann_msgs::msg::AckermannDriveStamped drive_msg;
        drive_msg.drive.steering_angle = steering_angle;
        drive_msg.drive.speed = get_speed(steering_angle, filtered_ranges, best_point_index);

        publisher_->publish(drive_msg);
    }
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutoDrive>());
    rclcpp::shutdown();
    return 0;
}
