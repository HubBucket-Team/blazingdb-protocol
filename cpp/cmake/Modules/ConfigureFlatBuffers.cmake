#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
#=============================================================================

include(ExternalProject)

ExternalProject_Add(google-flatbuffers_ep
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=RELEASE
    -DCMAKE_INSTALL_PREFIX=google-flatbuffers_prefix
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG v1.9.0
  UPDATE_COMMAND "")
ExternalProject_Get_property(google-flatbuffers_ep BINARY_DIR)
set(GOOGLE_FLATBUFFERS_ROOT ${BINARY_DIR}/google-flatbuffers_prefix)

file(MAKE_DIRECTORY ${GOOGLE_FLATBUFFERS_ROOT}/include)
file(MAKE_DIRECTORY ${GOOGLE_FLATBUFFERS_ROOT}/lib)

add_library(Google::Flatbuffers INTERFACE IMPORTED)
add_dependencies(Google::Flatbuffers google-flatbuffers_ep)
target_include_directories(Google::Flatbuffers
    INTERFACE ${GOOGLE_FLATBUFFERS_ROOT}/include)
target_link_libraries(Google::Flatbuffers
    INTERFACE ${GOOGLE_FLATBUFFERS_ROOT}/lib/libflatbuffers.a)

# Will define the target FlatBuffersSchemas and this can be used to reload/rebuild the schemas
#macro(DEFINE_FLATBUFFERSSCHEMAS_TARGET)
    ## This target will compile FlatBuffers schema files to *_generated.h headers and its binary format
    #set(target_name "FlatBuffersSchemas")
    #set(schema_dirs "${PROJECT_SOURCE_DIR}/../messages")

    ## build_flatbuffers arguments
    #set(flatbuffers_schemas "${schema_dirs}/*.fbs")
    #set(schema_include_dirs "")
    #set(custom_target_name "${target_name}")
    #set(additional_dependencies "")
    #set(generated_includes_dir "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}.dir/")
    #set(binary_schemas_dir "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}.dir/")
    #set(copy_text_schemas_dir "")

    #build_flatbuffers("${flatbuffers_schemas}" "${schema_include_dirs}" "${custom_target_name}" "${additional_dependencies}" "${generated_includes_dir}" "${binary_schemas_dir}" "${copy_text_schemas_dir}")

    ## This target will clear the generated sources fileas each time the user invoke the target FlatBuffersSchemas
    #add_custom_target(CleanFlatBuffersSchemas COMMAND rm -f *_generated.h && rm -f *.bfbs WORKING_DIRECTORY ${generated_includes_dir})
    #add_dependencies(FlatBuffersSchemas CleanFlatBuffersSchemas)

    #include_directories("${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${target_name}.dir")
#endmacro()

# Locate the FlatBuffers package.
# Requires that you build with:
#   -DFLATBUFFERS_HOME:PATH=/path/to/flatbuffers_install_dir
#message(STATUS "FLATBUFFERS_HOME: " ${FLATBUFFERS_HOME})
#find_package(FlatBuffers REQUIRED)
#set_package_properties(FlatBuffers PROPERTIES TYPE REQUIRED PURPOSE "FlatBuffers is an efficient cross platform serialization library." URL "https://google.github.io/flatbuffers/")

#if(NOT FLATBUFFERS_FOUND)
    #message(FATAL_ERROR "FlatBuffers not found, please check your settings.")
#endif()

#include_directories(${FLATBUFFERS_INCLUDE_DIR})
#include_directories(${FLATBUFFERS_ROOT}/include/)
#link_directories(${FLATBUFFERS_ROOT}/lib/)

#define_flatbuffersschemas_target()
