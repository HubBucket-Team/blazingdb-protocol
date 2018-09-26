#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#     Copyright 2018 Cristhian Alberto Gonzales Castillo <cristhian@blazingdb.com>
#=============================================================================

if (NOT GTEST_FOUND)

find_package(Threads)

include(GoogleTest)
include(ExternalProject)

ExternalProject_Add(googletest_ep
	CMAKE_ARGS
		-DCMAKE_BUILD_TYPE=RELEASE
		-DCMAKE_INSTALL_PREFIX=build
    EXCLUDE_FROM_ALL TRUE
	GIT_REPOSITORY git@github.com:google/googletest.git
    GIT_TAG release-1.8.1
	UPDATE_COMMAND "")
ExternalProject_Get_property(googletest_ep BINARY_DIR)
set(GTEST_ROOT ${BINARY_DIR}/build)

set(GTEST_FOUND ON)

file(MAKE_DIRECTORY ${GTEST_ROOT}/include)
file(MAKE_DIRECTORY ${GTEST_ROOT}/lib)

add_library(GTest::GTest INTERFACE IMPORTED)
add_dependencies(GTest::GTest googletest_ep)
target_include_directories(GTest::GTest INTERFACE ${GTEST_ROOT}/include)
target_link_libraries(GTest::GTest
    INTERFACE ${GTEST_ROOT}/lib/libgtest.a Threads::Threads)

add_library(GTest::Main INTERFACE IMPORTED)
add_dependencies(GTest::Main GTest::GTest)
target_link_libraries(GTest::Main INTERFACE ${GTEST_ROOT}/lib/libgtest_main.a)

enable_testing()

endif()
