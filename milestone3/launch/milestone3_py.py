#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='milestone3',
            namespace='milestone3',
            executable='depth_detection_node.py',
            name='depth_detection_node',
            parameters=[
                {'num_laps': 5}
            ]
        )
    ])
