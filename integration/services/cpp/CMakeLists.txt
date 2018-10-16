cmake_minimum_required(VERSION 3.11)
project(cpp-services)

set(CMAKE_CXX_STANDARD 11)
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
include(ConfigureCUDA)

include(ExternalProject)

ExternalProject_Add(blazingdb-protocol_ep
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=RELEASE
		-DCMAKE_INSTALL_PREFIX=blazingdb-protocol_prefix
	GIT_REPOSITORY git@github.com:BlazingDB/blazingdb-protocol.git
    GIT_TAG master
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

set(GOOGLE_FLATBUFFERS_ROOT
  ${BLAZINGDB_PROTOCOL_ROOT}/../google-flatbuffers_ep-prefix/src/google-flatbuffers_ep-build/google-flatbuffers_prefix)

file(MAKE_DIRECTORY ${GOOGLE_FLATBUFFERS_ROOT}/include)
file(MAKE_DIRECTORY ${GOOGLE_FLATBUFFERS_ROOT}/lib)

add_library(Google::Flatbuffers INTERFACE IMPORTED)
target_include_directories(Google::Flatbuffers
    INTERFACE ${GOOGLE_FLATBUFFERS_ROOT}/include)
target_link_libraries(Google::Flatbuffers
    INTERFACE ${GOOGLE_FLATBUFFERS_ROOT}/lib/libflatbuffers.a)

find_package( Threads )

set(LIBRARIES BlazingDB::Protocol Google::Flatbuffers  ${CMAKE_THREAD_LIBS_INIT} ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(blazingdb_orchestator_service src/orchestrator-service/orchestator-service.cu)
target_link_libraries(blazingdb_orchestator_service ${LIBRARIES} )

cuda_add_executable(blazingdb_ral_service src/ral-service/ral-server.cu)
target_link_libraries(blazingdb_ral_service ${LIBRARIES} )

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
