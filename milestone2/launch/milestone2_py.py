#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='milestone2',
            namespace='milestone2',
            executable='auto_drive_node.py',
            name='auto_drive_node'
        )
    ])
