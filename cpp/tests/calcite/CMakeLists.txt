set(CALCITE_DIR ${CMAKE_SOURCE_DIR}/src/blazingdb/protocol/calcite)

set(RelNodeBuilderTest_SRCS
  RelNodeBuilderTest.cpp
  utils.cpp
  ${CALCITE_DIR}/messages/RelNodeBuilder.cpp
)

configure_test(RelNodeBuilderTest "${RelNodeBuilderTest_SRCS}")
target_link_libraries(RelNodeBuilderTest Google::Flatbuffers)
target_include_directories(RelNodeBuilderTest PUBLIC "${CMAKE_SOURCE_DIR}/src/blazingdb/protocol/calcite/dto")

include(${CMAKE_SOURCE_DIR}/src/blazingdb/protocol/calcite/dto/CMakeLists.txt)
set(logical_plan_source_files ${source_files})
target_sources(RelNodeBuilderTest PUBLIC ${logical_plan_source_files})

add_subdirectory(dto)
