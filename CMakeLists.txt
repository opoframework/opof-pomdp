cmake_minimum_required(VERSION 3.14)
project(pomdp-core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "Require at least gcc-7.0")
endif()
SET(CMAKE_CXX_FLAGS "-fPIC -O2  -msse2 -mfpmath=sse ")


include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHON_NUMPY_INCLUDE_DIRS})

include_directories(pomdp_core/boost/include)
include_directories(pomdp_core/carla/include)
include_directories(pomdp_core/despot/include)
include_directories(pomdp_core/macaron/include)
include_directories(pomdp_core/rvo2/include)
include_directories(pomdp_core/magic/include)
include_directories(pomdp_core/opencv/include)

file(GLOB_RECURSE CARLA_SOURCES pomdp_core/carla/src/*.cpp)
file(GLOB_RECURSE DESPOT_SOURCES pomdp_core/despot/src/*.cpp)
file(GLOB_RECURSE MACARON_SOURCES pomdp_core/macaron/src/*.cpp)
file(GLOB_RECURSE RVO2_SOURCES pomdp_core/rvo2/src/*.cpp)
file(GLOB_RECURSE MAGIC_SOURCES pomdp_core/magic/src/*.cpp)
file(GLOB_RECURSE OPENCV_LIBS pomdp_core/opencv/lib/*.a)

# This lets us extract only what we need to not blow up the final size.
add_library(pomdp_core_base STATIC ${CARLA_SOURCES}
  ${DESPOT_SOURCES}
  ${MACARON_SOURCES}
  ${RVO2_SOURCES}
  ${MAGIC_SOURCES})
target_link_libraries(pomdp_core_base ${OPENCV_LIBS})
add_library(pomdp_core SHARED pomdp_core/pomdp_core.cpp)
target_link_libraries(pomdp_core pomdp_core_base ${OPENCV_LIBS})

set_target_properties(
  pomdp_core
  PROPERTIES PREFIX ""
             OUTPUT_NAME "pomdp_core"
             LINKER_LANGUAGE C)
