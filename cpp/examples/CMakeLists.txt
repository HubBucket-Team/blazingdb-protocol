cmake_minimum_required(VERSION 3.3)
project(blazing_protocol_examples)
find_package( Threads )

set(LIBRARIES blazingdb-protocol  ${CMAKE_THREAD_LIBS_INIT})

add_executable(blazingdb_orchestator_service src/orchestrator-service/orchestator-service.cc)
target_link_libraries(blazingdb_orchestator_service ${LIBRARIES})


add_executable(blazingdb_ral_service src/ral-service/ral-server.cc)
target_link_libraries(blazingdb_ral_service ${LIBRARIES})



add_executable(blazingdb_ral_client src/orchestrator-service/ral-client.cc)
target_link_libraries(blazingdb_ral_client ${LIBRARIES})

add_executable(api_client src/api/client-test.cpp)
target_link_libraries(api_client ${LIBRARIES})

add_executable(api_server src/api/server-test.cpp)
target_link_libraries(api_server ${LIBRARIES})