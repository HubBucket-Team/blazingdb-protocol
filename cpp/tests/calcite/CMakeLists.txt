set(CALCITE_DIR ${CMAKE_SOURCE_DIR}/src/blazingdb/protocol/calcite)

set(RelNodeBuilderTest_SRCS
  RelNodeBuilderTest.cpp
  utils.cpp
  ${CALCITE_DIR}/messages/RelNodeBuilder.cpp
)

configure_test(RelNodeBuilderTest "${RelNodeBuilderTest_SRCS}")
target_link_libraries(RelNodeBuilderTest Google::Flatbuffers)
