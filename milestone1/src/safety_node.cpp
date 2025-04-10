#include "rclcpp/rclcpp.hpp"
/// CHECK: include needed ROS msg type headers and libraries
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include <cmath>


class Safety : public rclcpp::Node {
// The class that handles emergency braking

public:
    Safety() : Node("safety_node")
    {
        publisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            drive_topic,
            10
        );
        scan_subscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidarscan_topic,
            10,
            std::bind(&AutoDrive::scan_callback, this, std::placeholders::_1)
        );
        odom_subscription = this->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            10,
            std::bind(&AutoDrive::odom_callback, this, std::placeholders::_1)
        );
    }

private:
    double speed;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_publisher;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription;

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
        // Update the current speed from the odometry message
        speed = msg->twist.twist.linear.x;
    }

    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
        const double threshold = 1.2; // Threshold for Time-to-Collision (TTC)
        auto ranges = scan_msg->ranges;
        const double angle_min = scan_msg->angle_min;
        const double angle_increment = scan_msg->angle_increment;

        for (size_t i = 0; i < ranges.size(); ++i) {
            double angle = angle_min + i * angle_increment;
            double v_long = speed * std::cos(angle);

            if (v_long > 0) {
                double iTTC = ranges[i] / v_long;
                if (iTTC < threshold) {
                    auto drive_msg = ackermann_msgs::msg::AckermannDriveStamped();
                    drive_msg.drive.speed = 0.0;
                    drive_publisher->publish(drive_msg);

                    RCLCPP_INFO(this->get_logger(), "EMERGENCY STOP: iTTC = %f", iTTC);
                    return;
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Safety>());
    rclcpp::shutdown();
    return 0;
}