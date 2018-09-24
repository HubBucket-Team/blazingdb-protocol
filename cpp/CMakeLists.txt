#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.3)
project(blazingdb-protocol)

set(CMAKE_CXX_STANDARD 11)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# Output all binaries and libs into a common folder across all libraries.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

###### OPTIONAL BUILDS
option(BUILD_TESTS "Build SIMPLE Tests" FALSE)
option(BUILD_EXAMPLES "Build SIMPLE Examples" FALSE)

# Configure dependencies
include(ConfigureFlatBuffers)
include(ConfigureGoogleTest)

# BEGIN MAIN #
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(SRC_FILES 
              ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/calcite_client.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/src/protocol/MessageBuilder.cpp
)

#add_executable(blazingdb-protocol ${SRC_FILES})

# STATIC library
add_library(blazingdb-protocol ${SRC_FILES})


# NOTE Each time we build the target blazingdb-protocol we will compile the flatbuffer schemas
add_dependencies(blazingdb-protocol FlatBuffersSchemas)


target_include_directories(${PROJECT_NAME} PUBLIC
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
set(BUILD_EXAMPLES True)

# Tests.
if (${BUILD_TESTS})
  include(CTest)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE MATCHES "Coverage") # Coverage flags for GCC
    add_compile_options(-g -O0 -fprofile-arcs -ftest-coverage)
    set(coverage_lib gcov)
  endif()
  enable_testing()
  add_subdirectory(tests)
endif()

# Examples.
if (${BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()


# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #