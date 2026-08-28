cmake_minimum_required(VERSION 3.16)
project(phylo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)          # optional: replaces mclapply parallelism

add_library(phylo_core
  src/rng.cpp src/tree.cpp src/clades.cpp src/transition.cpp
  src/pruning.cpp src/csv.cpp src/config.cpp src/init.cpp
  src/gibbs.cpp src/topology.cpp src/smc.cpp)

target_include_directories(phylo_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external/eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(phylo_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(MERRCh src/main.cpp)
target_link_libraries(MERRCh PRIVATE phylo_core)


