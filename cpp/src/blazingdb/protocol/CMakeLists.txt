include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(SRCS
  client/client.cc
  connection/unix_socket_connection.cc
  server/server.cc
  utilities/io_util.cc
  message/interpreter/utils.cpp
  message/interpreter/messages.cpp
  message/messages.cpp
  message/io/file_system.cpp
)
find_package(Threads)

add_library(blazingdb-protocol ${SRCS})
target_link_libraries(blazingdb-protocol Threads::Threads  ${ZEROMQ_STATIC_LIB} ${FLATBUFFERS_STATIC_LIB})

add_custom_target("generate-flatbuffers-messages")
add_custom_command(TARGET "generate-flatbuffers-messages"
  COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} -c --gen-all
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

add_dependencies(blazingdb-protocol "generate-flatbuffers-messages")

if (GTEST_FOUND)
  add_executable(server-test EXCLUDE_FROM_ALL server-test.cc)
  target_link_libraries(server-test blazingdb-protocol GTest::GTest GTest::Main)
  gtest_add_tests(TARGET server-test)
endif()
