#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Here is an example of how to launch a node
        # Node(
        #     package='your_package_name',
        #     namespace='your_namespace', # change if using multiple nodes of same type
        #     executable='executable_name', # if using python, append .py to the end; for cpp you don't need to do anything
        #     name='executable_name',
        #     parameters=[
        #         {'parameter_name1': value1},
        #         {'parameter_name2': value2}
        #     ]
        # ),
        # this one for the python implementation
        Node(
            package='milestone1',
            namespace='milestone1',
            executable='auto_drive_node.py',
            name='auto_drive_node'
        )
    ])
