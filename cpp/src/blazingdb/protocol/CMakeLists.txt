set(SRCS
  client/client.cc
  connection/unix_socket_connection.cc
  server/server.cc
)

add_library(blazingdb-protocol ${SRCS})

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

if (GTEST_FOUND)

add_executable(server-test EXCLUDE_FROM_ALL server-test.cc)
target_link_libraries(server-test blazingdb-protocol GTest::GTest GTest::Main)
gtest_add_tests(TARGET server-test)

endif()
