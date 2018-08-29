cmake_minimum_required(VERSION 2.8.3)
project(quad_control)
find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs asctec_hl_comm)

catkin_package()
include_directories( include $catkin_INCLUDE_DIRS})

add_executable(convert src/convert.cpp)
target_link_libraries(convert ${catkin_LIBRARIES})
