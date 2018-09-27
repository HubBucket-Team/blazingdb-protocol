cmake_minimum_required(VERSION 3.3)


set(CMAKE_CXX_STANDARD 11)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

# Configure dependencies
include(ConfigureFlatBuffers)

include(ExternalProject)

ExternalProject_Add(blazingdb-protocol_ep
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=RELEASE
		-DCMAKE_INSTALL_PREFIX=blazingdb-protocol_prefix
	GIT_REPOSITORY git@github.com:BlazingDB/blazingdb-protocol.git
    GIT_TAG develop
    SOURCE_SUBDIR cpp
	UPDATE_COMMAND "")
ExternalProject_Get_property(blazingdb-protocol_ep BINARY_DIR)
set(BLAZINGDB_PROTOCOL_ROOT ${BINARY_DIR}/blazingdb-protocol_prefix)

file(MAKE_DIRECTORY ${BLAZINGDB_PROTOCOL_ROOT}/include)
file(MAKE_DIRECTORY ${BLAZINGDB_PROTOCOL_ROOT}/lib)

add_library(BlazingDB::Protocol INTERFACE IMPORTED)
add_dependencies(BlazingDB::Protocol blazingdb-protocol_ep)
target_include_directories(BlazingDB::Protocol
    INTERFACE ${BLAZINGDB_PROTOCOL_ROOT}/include)
target_link_libraries(BlazingDB::Protocol
    INTERFACE ${BLAZINGDB_PROTOCOL_ROOT}/lib/libblazingdb-protocol.a)

add_executable(blazingdb-orchestator src/blazingdb-orchestator.cc)

target_link_libraries(blazingdb-orchestator BlazingDB::Protocol FlatBuffersSchemas)
