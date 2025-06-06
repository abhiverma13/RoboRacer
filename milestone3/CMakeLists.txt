cmake_minimum_required(VERSION 3.5)
project(milestone3)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
foreach(PACKAGE 
  ament_cmake 
  ament_cmake_python
  rclcpp
  rclpy 
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Include C++ "include" directory
include_directories(include)

# Create C++ executables
# Repeat this step for each C++ executable
add_executable(auto_drive_node src/auto_drive_node.cpp)
ament_target_dependencies(auto_drive_node 
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs
)

# Install C++ executables
# Add a new line for each C++ executable
install(TARGETS
  auto_drive_node
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
# Add a new line for each Python executable
install(PROGRAMS
  scripts/auto_drive_node.py
  scripts/camera_drive_node.py
  scripts/depth_detection_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



ament_package()
