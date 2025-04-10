#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='milestone4',
            namespace='milestone4',
            executable='auto_drive_node_fast.py',
            name='auto_drive_node',
            parameters=[
                {'num_laps': 7}
            ]
        )
    ])
