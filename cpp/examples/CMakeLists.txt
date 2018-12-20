cmake_minimum_required(VERSION 3.12)
project(blazing_protocol_examples)
find_package(Threads)

set(LIBRARIES Threads::Threads blazingdb-protocol)

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(api_client src/api/client-test.cpp)
target_link_libraries(api_client ${LIBRARIES})

add_executable(api_server src/api/server-test.cpp)
target_link_libraries(api_server ${LIBRARIES})


add_executable(io_server src/io/server-test.cpp)
target_link_libraries(io_server ${LIBRARIES})

add_executable(blazingdb_orchestator_service src/orchestrator-service/orchestator-service.cc)
target_link_libraries(blazingdb_orchestator_service ${LIBRARIES})
 