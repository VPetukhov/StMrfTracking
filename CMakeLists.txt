cmake_minimum_required(VERSION 2.8)
project(StMrf)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# GCO
set(GCO_SRC
        gco-v3.0/GCoptimization.cpp
        gco-v3.0/GCoptimization.h
        gco-v3.0/LinkedBlockList.cpp
        gco-v3.0/LinkedBlockList.h
        gco-v3.0/block.h
        gco-v3.0/energy.h
        gco-v3.0/graph.h)
add_library(gco ${GCO_SRC})

# StMRF
#find_package(OpenCV REQUIRED)
set(INCLUDE_DIRS /home/viktor/local/anaconda3/include gco-v3.0/)
include_directories(${INCLUDE_DIRS})

FILE(GLOB OpenCV_LIBRARIES /home/viktor/local/anaconda3/lib/libopencv_*.so)
message(${OpenCV_LIBRARIES})

set(CMAKE_CXX_STANDARD 11)

FILE(GLOB StMrfTrackingSources Tracking/*.cpp)
add_library(StMrfTracking ${StMrfTrackingSources})
target_link_libraries(StMrfTracking ${OpenCV_LIBRARIES} gco)

add_executable(StMrf main.cpp)
target_link_libraries(StMrf StMrfTracking ${OpenCV_LIBRARIES} gco)