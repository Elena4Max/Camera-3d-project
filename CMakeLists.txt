cmake_minimum_required(VERSION 3.16)
project(Camera3D)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

add_executable(app
    src/main.cpp
    src/calibration/calibration.cpp
    src/detector/yolo.cpp
    src/geometry/homography.cpp
    src/geometry/projection.cpp
)

target_link_libraries(app ${OpenCV_LIBS})
