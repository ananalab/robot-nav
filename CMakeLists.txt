cmake_minimum_required(VERSION 3.20)
project(robot_nav LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV QUIET)
set(WITH_OPENCV OFF)
if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER_EQUAL 4.7)
  set(WITH_OPENCV ON)
else()
  message(WARNING "OpenCV >= 4.7 absent : seuls la simulation physique et bench sont compilés")
endif()

# Algorithmes (sans OpenCV) : core, planification, contrôle, estimation, simulation, liaison série.
add_library(rn src/core.cpp src/planning.cpp src/control.cpp src/estimation.cpp src/sim.cpp src/hw.cpp)
target_include_directories(rn PUBLIC src firmware/robot_fw)
target_link_libraries(rn PUBLIC Eigen3::Eigen yaml-cpp::yaml-cpp Threads::Threads)

# Navigation + épisodes de simulation ; avec OpenCV : vision et outils en plus.
if(WITH_OPENCV)
  add_library(rn_app src/app.cpp src/vision.cpp src/tools.cpp)
  target_compile_definitions(rn_app PUBLIC RN_WITH_OPENCV)
  target_include_directories(rn_app SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(rn_app PUBLIC rn ${OpenCV_LIBS})
