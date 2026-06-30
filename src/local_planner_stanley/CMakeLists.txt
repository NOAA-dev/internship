cmake_minimum_required(VERSION 3.8)
project(local_planner_stanley)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(custom_interfaces REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(Stanley src/stanley_for_sim.cpp)
ament_target_dependencies(Stanley rclcpp std_msgs nav_msgs tf2_ros tf2_geometry_msgs custom_interfaces geometry_msgs)

add_executable(Stanley_real src/stanley_for_real_bot.cpp)
ament_target_dependencies(Stanley_real rclcpp std_msgs nav_msgs tf2_ros tf2_geometry_msgs custom_interfaces geometry_msgs)

install(TARGETS
        Stanley
        Stanley_real
        DESTINATION lib/${PROJECT_NAME}
)

ament_package()
