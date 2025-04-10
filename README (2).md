# Milestone 3: On-Car Reactive Driving using Computer Vision

## Contents

- [Algorithms](#algorithms)
- [Testing Strategies](#testing-strategies)
- [Improvements over Milestones 1 and 2](#improvements-over-milestones-1-and-2)
- [Usage](#usage)
- [Final Thoughts](#final-thoughts)

## Algorithms

For Milestone 3, we transitioned from using LIDAR-based navigation to a computer vision-based approach, leveraging depth images and lane detection for reactive driving. The implementation uses a PID-controlled steering mechanism based on visual lane and depth data.

### Chosen Algorithm - Depth-Based Steering

- The system processes depth images from the car’s camera to identify the furthest point within a defined detection box.
- A PID controller is used to compute the steering angle based on the horizontal error between the detected furthest point and the center of the detection box.
- The speed is adjusted dynamically based on the steering angle, slowing down for sharp turns.
- Watch a demo [here](https://drive.google.com/file/d/1XW31VTOM74LNyIkdMS43W8ConVA_eIzy/view).

### Tested and Unsuccessful Algorithm - Lane Detection and Path Following

- The system extracts lane features from RGB camera images using HSV thresholding and morphological operations.
- It detects lane boundaries by scanning for the largest consecutive white pixel segments.
- A polynomial fit is used to determine the mid-lane trajectory.
- The car’s center offset is computed relative to the detected lane, and a PID controller adjusts the steering angle accordingly.
- Watch a demo [here](https://drive.google.com/file/d/19iW2h2kD771bI_o77ZmwB_xJ6wCiiBrG/view). Blue is where we want to go, yellow is the furthest point detected, and green is where on the blue line we measure the turn from.

### Additional Algorithm - Emergency Stop Mechanism

- An emergency stop system ensures that if the car gets too close to an obstacle, it halts all movement.
- Based off the code for `safety_node`.

## Testing Strategies

### Testing the Lane Detection Algorithm

- The lane detection algorithm implementation was first tested in a simulation environment.
- After enough parameter tuning with the speed, the PID coefficients, and the steering angles, the car started driving smoothly in the simulation, and thus we moved to testing it in a physical environment.
- In the physical environment, we found that the lane detection algorithm induced a lot of real-time lag, meaning that it would publish values a significant amount of time after they were due.
- We tweaked around with trying to optimise the algorithm to reduce this latency, however we found that perhaps transitioning to a different algorithm would be the best call of action.

### Testing the Depth-Based Algorithm

- The depth-based algorithm was initially tested in simulation using different track layouts to evaluate its responsiveness to depth variations and path-following accuracy.
- We experimented with different PID parameters (`Kp`, `Ki`, `Kd`) to minimize oscillations and improve turn handling.
- The detection box size and position were adjusted to optimize how the system identifies the furthest point in the scene.
- Once the algorithm performed well in simulation, we transitioned to real-world testing.
- In real-world tests, the system demonstrated improved response time and steering accuracy compared to the lane detection approach.
- We noticed that the car would go quite close to the wall on the bends, and hence we tried increasing the speed a bit and tweaking the steering coefficients.
- Once we found the car to reliably turn the corners, we optimised for speed to reduce the lap time.
- The final implementation showed a significant reduction in latency compared to the lane detection algorithm, making it more reliable for real-time driving conditions.

## Improvements over Milestones 1 and 2

1. **Transition to Computer Vision**

   - Replaced LIDAR-based gap-following with depth image processing for navigation.

2. **Enhanced Steering Control**

   - Implemented PID-based steering correction based on both depth and lane data.
   - Adjusted vehicle speed dynamically based on turn sharpness.

3. **More Robust Obstacle Avoidance**

   - Integrated an emergency stop system for added safety.
   - Improved detection reliability with a combination of depth and lane data.

4. **Better Adaptability to Track Conditions**

   - Adjusted camera-based vision algorithms for noise resilience.

## Usage

To set this up on the car, do this once:

- `cd` into the sim workspace where this repo is under the `/src` directory.
- Run `colcon build --packages-select milestone3`.
- Run `source install/local_setup.bash`.

To run the nodes whenever, do this:

- Run `ros2 launch realsense2_camera rs_launch.py`
- In another terminal, either
   - (a) run the node directly by executing `ros2 run milestone3 depth_detection_node.py`
   - (b) run the node using the launch file by executing `ros2 launch milestone3 milestone3_py.py`
   - or (c) run the python file from VS Code by pressing the `Run Python File` button or by executing `python3 <path>/<to>/depth_detection_node.py`. Make sure the python file has execute permissions.

The number of laps can also be set while the nodes are running via the command `ros2 param set /depth_detection_node num_laps <VALUE>`, where \<VALUE\> is an integer.

If you are missing any dependencies or packages, you can install them separately using `pip install <PACKAGE_NAME>` or by using `pip install -r <path>/<to>/Requirements.txt`.

## Final Thoughts

Overall, the camera was a larger pain compared to using the LiDAR. While there were libraries to use for the computer vision stuff, the code for this milestone was more complex compared to milestone 2 due to all the processing that had to be done, which in turn added a lot more parameters to tweak.

Additionally, a third of our members were not able to run the AutoDRIVE Simulator due to hardware issues. So, we did a lot of testing using video/image captures taken by the car's camera at first. Although the [simulation](https://drive.google.com/file/d/1C6dqF2A6V0NOGl2KTBqJQbVaQVHaKFd4/view) ran okay, this *did not* translate over to the physical car at all. There were so many parameters to test that we had to be in the lab for.

It may be because we were using Python, but we noticed the camera-based algorithm required a lot more compute than our LiDAR-based algorithm and did not perform nearly as well. Moving forward, we will likely not use the camera as a main tool for navigation, but rather for object detection and auxiliary lane navigation in addition to the LiDAR.
