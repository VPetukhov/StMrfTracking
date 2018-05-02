cmake_minimum_required(VERSION 3.9)
project(StMrf)

#find_package(OpenCV REQUIRED)
set(INCLUDE_DIRS /home/viktor/local/anaconda3/include)
include_directories(${INCLUDE_DIRS})

FILE(GLOB OpenCV_LIBRARIES /home/viktor/local/anaconda3/lib/libopencv_*.so)
message(${OpenCV_LIBRARIES})

set(CMAKE_CXX_STANDARD 11)

FILE(GLOB StMrfTrackingSources Tracking/*.cpp)
add_library(StMrfTracking ${StMrfTrackingSources})
target_link_libraries(StMrfTracking ${OpenCV_LIBRARIES})

add_executable(StMrf main.cpp)
target_link_libraries(StMrf StMrfTracking ${OpenCV_LIBRARIES})