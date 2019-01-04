cmake_minimum_required(VERSION 3.12)
project(blazing_protocol_examples)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# NOTE Do not change the inclusion order
# Configure dependencies
include(ConfigureFlatBuffers)
include(ConfigureBlazingDBProtocol)
include(ConfigureZeroMQ)

find_package(Threads)
find_package(ZeroMQ REQUIRED)


# BEGIN MAIN #
include_directories(
    ${CMAKE_SOURCE_DIR}/src
)

set(LIBRARIES Threads::Threads blazingdb-protocol libzmq-static)

add_executable(api_client src/api/client-test.cpp)
target_link_libraries(api_client ${LIBRARIES})

add_executable(api_server src/api/server-test.cpp)
target_link_libraries(api_server ${LIBRARIES})

add_executable(io_server src/io/server-test.cpp)
target_link_libraries(io_server ${LIBRARIES})

add_executable(blazingdb_orchestator_service src/orchestrator-service/orchestator-service.cc)
target_link_libraries(blazingdb_orchestator_service ${LIBRARIES})
 