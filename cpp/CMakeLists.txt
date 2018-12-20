#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
#     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.11)

project(blazingdb-protocol)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

# TODO percy felipe alexander remove these lines when cudf stop using old C++ ABI
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#message(AUTHOR_WARNING "Using old C++ ABI: -D_GLIBCXX_USE_CXX11_ABI=0")

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
option(SIMPLE_BUILD_TESTS "Build SIMPLE Tests" FALSE)
option(SIMPLE_BUILD_EXAMPLES "Build SIMPLE Examples" FALSE)

# Configure dependencies
include(ConfigureFlatBuffers)
include(ConfigureGoogleTest)

# BEGIN MAIN #
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

## NOTE Each time we build the target blazingdb-protocol we will compile the flatbuffer schemas
# add_dependencies(blazingdb-protocol FlatBuffersSchemas)

# Tests.
if (${SIMPLE_BUILD_TESTS})
  include(CTest)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_BUILD_TYPE MATCHES "Coverage") # Coverage flags for GCC
    add_compile_options(-g -O0 -fprofile-arcs -ftest-coverage)
    set(coverage_lib gcov)
  endif()
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(src)

# Examples.
if (${SIMPLE_BUILD_EXAMPLES})
  add_subdirectory(examples)
endif()

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
