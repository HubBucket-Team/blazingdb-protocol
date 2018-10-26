cmake_minimum_required(VERSION 3.3)
project(blazing_protocol_examples)
include(ConfigureCUDA)
find_package(Threads)

set(LIBRARIES blazingdb-protocol  ${CMAKE_THREAD_LIBS_INIT})

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(api_client src/api/client-test.cpp)
target_link_libraries(api_client ${LIBRARIES})

add_executable(api_server src/api/server-test.cpp)
target_link_libraries(api_server ${LIBRARIES})


cuda_add_executable(blazingdb_orchestator_service src/orchestrator-service/orchestator-service.cc)
target_link_libraries(blazingdb_orchestator_service ${LIBRARIES})

cuda_add_executable(blazingdb_ral_client src/orchestrator-service/ral-client.cu)
target_link_libraries(blazingdb_ral_client ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(blazingdb_ral_service src/ral-service/ral-server.cu)
target_link_libraries(blazingdb_ral_service ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(ipc_server src/ipc_py.cu)
target_link_libraries(ipc_server ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(ipc_numba src/ipc_numba.cu)
target_link_libraries(ipc_numba ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(table_group src/table_group-demo.cc)
target_link_libraries(table_group ${LIBRARIES} )
