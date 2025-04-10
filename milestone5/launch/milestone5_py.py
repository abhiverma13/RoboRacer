#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='milestone5',
            namespace='milestone5',
            executable='auto_drive_node_fast.py',
            name='auto_drive_node',
            parameters=[
                {'num_laps': 7}
            ]
        )
    ])
