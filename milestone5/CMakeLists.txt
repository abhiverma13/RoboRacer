cmake_minimum_required(VERSION 3.5)
project(milestone5)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  # set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD 17) # changed to 17 for milestone5 and onwards
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

# OpenCV stuff
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Include C++ "include" directory
include_directories(include)

# Create C++ executables
# Repeat this step for each C++ executable
add_executable(auto_drive_node src/auto_drive_node.cpp)
ament_target_dependencies(auto_drive_node 
  rclcpp geometry_msgs ackermann_msgs nav_msgs sensor_msgs std_msgs cv_bridge
)

target_link_libraries(auto_drive_node ${OpenCV_LIBS})

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
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/auto_drive_node_fast.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/auto_drive_node_slow.py
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