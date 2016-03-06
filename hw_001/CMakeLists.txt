cmake_minimum_required(VERSION 2.8.4)

project(hw_001)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)
endif()
# find_package(Eigen 3 REQUIRED)


include_directories(include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(estimate_voltage
	src/estimate_voltage.cc
)

target_link_libraries(estimate_voltage
  ${Boost_LIBS}
  # ${Eigen_LIBS}
  ${OpenCV_LIBRARIES}
)
