# Milestone 2: On-Car Reactive Driving with LIDAR

## Demo
[Video Link](https://drive.google.com/file/d/1kl4VLGwx1aqDtf6NbnDvkB67zsDYeCL6/view?usp=sharing)

## Contents

* [Gap Follow Algorithm](#gap-follow-algorithm)
* [Avoiding Collisions](#avoiding-collisions)
* [Testing Methodology](#testing-methodology)
* [Improvements Over Milestone 1](#improvements-over-milestone-1)

## Gap Follow Algorithm

For Milestone 2, we transitioned from a wall-following approach to a gap-follow algorithm to improve driving performance, especially in environments with sharp turns and obstacles. This is due to the fact that when we tried our wall-following algorithm on the physical car, it had significant limitations that we difficult to overcome without an algorithm overhaul (such as obstacle avoidance).

### Algorithm Overview

The algorithm follows these main steps:
1. **Preprocess LIDAR Scan**: We take LIDAR scans and filter out invalid or out-of-range readings. A smoothing function is applied using a sliding window to reduce noise.
2. **Identify Safety Bubble**: The closest obstacle is identified, and a safety bubble is created around it to ensure the vehicle avoids collision.
3. **Detect Maximum Gap**: The largest open space in the LIDAR scan is identified by finding continuous valid readings.
4. **Determine Best Steering Point**: The midpoint of the largest gap is selected as the optimal driving direction.
5. **Apply PID Control**: A PID controller calculates the steering angle based on the error between the selected gap center and the vehicle’s heading.
6. **Speed Adjustment**: Speed is dynamically adjusted based on steering angle to maintain stability at high speeds.

### LIDAR Processing

The LIDAR scan is truncated to focus only on relevant data, using a predefined angle range. The algorithm then processes the scan to remove NaN values and applies a smoothing filter to enhance reliability.

### Safety Bubble

A safety bubble is drawn around the nearest obstacle by setting the LIDAR readings within a certain radius to zero, preventing the vehicle from choosing a path too close to the obstacle.

### Finding the Largest Gap

The algorithm scans for the longest sequence of valid LIDAR readings. Once found, the midpoint of the gap is selected as the optimal driving path.

### PID Controller

We modified our PID controller from Milestone 1, instead of the PID controller setting the speed of the vehicle, now the PID sets the steering angle based on the error.

### Speed Control

Speed is dynamically set based on the magnitude of the steering angle:
- **Small angles** → higher speeds (e.g., 3.0 m/s)
- **Larger angles** → lower speeds (e.g., 0.5 m/s)

Speed is also increased dynamically based on the distance in front of the car using an exponential function:
- **equation** → e^(0.04 distance)
 
This helps maintain vehicle stability while navigating sharp turns.

## Avoiding Collisions

The algorithm incorporates automatic braking and collision avoidance:
- The car slows down when the steering angle changes.
- The safety bubble mechanism ensures that the vehicle does not attempt to drive into obstacles.

## Testing Methodology

### Simulation-Based Testing

The initial implementation was tested in simulation:
1. **Basic Movement**: Ensuring the vehicle follows an open gap without oscillations.
2. **Sharp Turns**: Verifying that the car adjusts correctly when encountering sharp turns.
3. **Obstacle Avoidance**: Ensuring the vehicle correctly avoids obstacles by selecting alternate paths.

### Physical Parameter Tuning

Once we had our implementation working in the simulations, we tested it out on the actual car.

PID tuning followed these steps:
- We started with `k_p = 0.6`, `k_d = 0.3`, and `k_i = 0.00001` from milestone 1.
- After testing the car on the physical track, we determined that the `k_d` value needed to be reduced, and we applied trial and error until we settled for `k_d = 0.15`.
- We also slightly tweaked the `k_p` value to `k_p = 0.65`.
  
We found that the parameters that needed to be tuned the most were the bubble radius, which determined how much of the area around the closest point would be set to 0, and the speeds, which were based on the steering angles.
- A bubble radius that was too high (0.5m) would lead to overdetection too much turning. A bubble radius too low (0.25m) would lead to the car not detecting the turns quick enough and crashes.
- We settled on a bubble radius of 0.6m.
- For the speeds, we started very slow (0.8m/s) for the straights (when our car had a low steering angle) and slowly incremented up. For the turns we kept it relatively slow, when we tried increasing the speed it led to collisions.

## Improvements Over Milestone 1

1. **Switched to Gap Follow Algorithm**:
   - Wall-following was unreliable in sharp turns and obstacle-heavy environments.
   - The new algorithm enables better maneuverability in diverse conditions.

2. **LIDAR Processing**:
   - Improved filtering and smoothing for more accurate distance measurements.
   - Dynamic gap selection instead of static wall measurements.

3. **More Robust Collision Avoidance**:
   - The vehicle no longer stops abruptly but adjusts its path dynamically.
   - The safety bubble ensures obstacle avoidance while maintaining movement.

4. **Better Speed Control**:
   - Speed dynamically adjusts based on steering angle, improving stability.
   - Higher speeds on straight paths and lower speeds on sharp turns.
   - Implemented a distance-based speed boost, eg: when the distance is large, a speed boost applies. When the distance is small, a little speed boost applies.

With these improvements, our reactive driving approach is significantly more adaptable and efficient than the previous milestone’s wall-following method.

