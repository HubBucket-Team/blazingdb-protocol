set(SRCS
  client/client.cc
  connection/unix_socket_connection.cc
  server/server.cc
)

cuda_add_library(blazingdb-protocol ${SRCS})
target_link_libraries(blazingdb-protocol Google::Flatbuffers)

add_custom_command(TARGET blazingdb-protocol
  COMMAND ${GOOGLE_FLATBUFFERS_ROOT}/bin/flatc -c --gen-all
    ${PROJECT_SOURCE_DIR}/../messages/all.fbs
  COMMENT "Generate all messages")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/all_generated.h
  DESTINATION include/blazingdb/protocol)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/blazingdb
  FILES_MATCHING PATTERN "*.h")

install(TARGETS blazingdb-protocol
  ARCHIVE
    DESTINATION lib
    COMPONENT API
  LIBRARY
    DESTINATION lib
    COMPONENT API
  PUBLIC_HEADER
    DESTINATION include
    COMPONENT API)

# add_dependencies(blazingdb-protocol FlatBuffersSchemas)

if (GTEST_FOUND)
  add_executable(server-test EXCLUDE_FROM_ALL server-test.cc)
  target_link_libraries(server-test blazingdb-protocol GTest::GTest GTest::Main)
  gtest_add_tests(TARGET server-test)
endif()
