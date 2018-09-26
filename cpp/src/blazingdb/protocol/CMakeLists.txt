set(SRCS api.cc)

add_library(blazingdb-protocol ${SRCS})

install(FILES api.h DESTINATION include/blazingdb/protocol)

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
target_link_libraries(server-test GTest::GTest GTest::Main)
gtest_add_tests(TARGET server-test)

endif()
