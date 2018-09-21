cmake_minimum_required(VERSION 3.3)
project(blazing_protocol_examples)

set(LIBRARIES blazingdb-protocol)

add_executable(example_client src/client-example.cpp)
target_link_libraries(example_client ${LIBRARIES})

add_executable(example_server src/server-example.cpp)
target_link_libraries(example_server ${LIBRARIES})

# install(TARGETS example_client example_server DESTINATION ${CMAKE_INSTALL_BINDIR})
