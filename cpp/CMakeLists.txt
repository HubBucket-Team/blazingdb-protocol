#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#     Copyright 2018 Alexander Ocsa <alexander@blazingdb.com>
#     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
#=============================================================================

cmake_minimum_required(VERSION 3.11)

project(blazingdb-protocol)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# NOTE Import cmake modules from blazingdb-dependencies (Mandatory: First step!)
include(ImportCMakeModules)
include(ConfigureBlazingDBHostCompiler)

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# NOTE Configure dependencies (always include this one first)
include(ConfigureBlazingDBDependencies)

# Output all binaries and libs into a common folder across all libraries.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

###### OPTIONAL BUILDS
option(SIMPLE_BUILD_TESTS "Build SIMPLE Tests" FALSE)

# Configure dependencies
include(ConfigureFlatBuffers)
include(ConfigureGoogleTest)

# BEGIN MAIN #

# TODO percy move this to tool-chain and add support for cuda def and more than 1 defs
add_definitions(${CXX_DEFINES})
get_directory_property(CXX_COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
message(STATUS "C++ compiler definitions: ${CXX_COMPILE_DEFINITIONS}")

find_package(Threads)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/src)

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

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)

#END MAIN #
