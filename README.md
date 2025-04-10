# Milestone 5: 1 VS 1 Racing

## Contents

- [Algorithms](#algorithms)
- [Testing Strategies](#testing-strategies)
- [Improvements over Previous Milestones](#improvements-over-previous-milestones)
- [Usage](#usage)
- [Final Thoughts](#final-thoughts)

## Algorithms

For Milestone 5, our driving algorithm is almost the same from Milestone 4 - a reactive driving algorithm primarily using the car's LiDAR for the main navigation logic, however, we have tuned parameters further to improve speed. We have also edited our safety stops to slow down and avoid an object (or another car) if it is in front of our car or come to a complete stop if an object (or another car) gets too close to the car. The final algorithm that we used for the milestone 4 evaluation can be found in `milestone5/scripts/auto_drive_node_fast.py`.

### Speed Optimization

- Improved the `get_speed` function to use an exponential speed adjustment based on the car’s proximity to obstacles on low turning angles.
- Increased the base speed and other speeds compared to Milestone 4 while ensuring smoother acceleration and deceleration to prevent jerky movements.
- Heavily tested and adjusted PID tuning in `pid_control` function to balance aggressive speed control with stability.

### Overtaking

- Introduced a moving history of gap distances to make more intelligent path selection decisions, preventing sudden, unnecessary direction changes.
- In `preprocess_lidar_scan`, calculated horizontal distance of every LIDAR point and if distance is within 15cm (half of width of car) and LIDAR distance is measured to be less than 1.5m then zero out the point (similar idea to `draw_safety_bubble` function). This allows for faster object detection for objects that in front of the car ensures the car will not drive towards them.
- Added more angle processesing after PID controller in `pid_control` function to make car drive smoother on straights and take sharper turns when detecting obstacles.
- Improved `draw_safety_bubble` by adjusting the safety radius based on the car’s speed, ensuring safer navigation in high-speed scenarios.
- Heavily tested the car’s ability to anticipate dynamic obstacles and smoothly re-route while maintaining a reasonable speed.

### Enhanced Safety Mechanisms

- Edited `new_safety_stop` function, which finds the average of distances in the gap found by `find_max_gap` over the last 20 gaps and slows down if this average is less than the safety stop threshold. This ensures the car will slow down if there is no path to be taken. It also finds average width of the gap and if this width is less than a threshold, the car will also slow down, ensuring the car does not attempt to navigate through overly narrow gaps.
- The safety stop threshold is changed dynamically based on the car's average speed over the last 10 speeds.
- `safety_stop` is used to fully stop the car if any object gets too close to the car.

## Testing Strategies

### Speed and Turning Angle Testing

- Conducted multiple 1v1 race trials using various PID configurations to fine-tune acceleration, braking, and stability.
- Compared the updated `get_speed` function against the Milestone 4 implementation, observing a roughly **15% improvement** in average lap speed while maintaining control and safety.
- Evaluated the interaction between the new speed control and turning angle logic to ensure the vehicle maintained smooth navigation without overcorrection at high speeds.

### Overtaking Testing

- Used dynamic obstacles (by moving box around with broom) at key points on the track to evaluate overtaking algorithm.

### Safety Testing

- Used dynamic obstacles (by moving box around with broom) and static obstacles to slow down and fully stop car if there is no path to be taken.
- Rigorously tested emergency stop scenarios using varied obstacle types and placements to ensure activation only under necessary conditions.

## Improvements Over Previous Milestones

1. **Higher Speed with Enhanced Stability**
   - Achieved an average lap time of 8.2s
   - Upgraded `get_speed` to allow for a higher base speed while preserving vehicle control.
   - Fine-tuned acceleration and braking curves for smoother transitions during speed changes.

2. **Advanced Overtaking/Obstacle Avoidance**
   - Refined path selection to eliminate erratic turns and unnecessary maneuvers.
   - Improved `draw_safety_bubble` to scale more effectively with speed, enhancing obstacle detection in fast-paced situations.
   - Increased accuracy in detecting and responding to dynamic obstacles using both LIDAR and gap history data.

3. **Robust Safety Enhancements**
   - Improved emergency stop logic to act only when required, reducing unintended halts.
   - Introduced a history-based filtering mechanism in safety stop routines to better distinguish real threats from noise.

## Usage

To set this up on the car or simulator, do this once:

- `cd` into the workspace where this repo is under the `/src` directory.
- Run `colcon build --packages-select milestone5`.
- Run `source install/local_setup.bash`.

If running in the simulator:

- Make sure you've properly installed and set up the simulator.
- In one terminal session, run `ros2 launch f1tenth_gym_ros gym_bridge_launch.py`.
- In another terminal session, run `ros2 run milestone5 auto_drive_node_fast.py`.
- If you want to run the simulation with two cars instead, change `num_agent` to 2 in the sim.yaml file, and run `ros2 launch milestone5 milestone5_py.py`.

If running on the car:

- In a terminal, run the node by executing `ros2 run milestone5 auto_drive_node_fast.py --ros-args -p num_laps:YOUR_VALUE_HERE` where `YOUR_VALUE_HERE` is the amount of laps you want the car to run (set it to -1 if you would like it to lap indefinitely).

## Final Thoughts

Overall, we are content with the results. We managed to create an algorithm that allows our car to drive at higher speeds, navigating through more difficult obstacles, and slow down or stop safely in time when an emergency occurs. 