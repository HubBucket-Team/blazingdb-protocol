cmake_minimum_required(VERSION 3.3)
project(blazing_protocol_examples)
include(ConfigureCUDA)
find_package(Threads)

set(LIBRARIES blazingdb-protocol  ${CMAKE_THREAD_LIBS_INIT})

add_executable(api_client src/api/client-test.cpp)
target_link_libraries(api_client ${LIBRARIES})

add_executable(api_server src/api/server-test.cpp)
target_link_libraries(api_server ${LIBRARIES})


cuda_add_executable(blazingdb_orchestator_service src/orchestrator-service/orchestator-service.cu)
target_link_libraries(blazingdb_orchestator_service ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(blazingdb_ral_client src/orchestrator-service/ral-client.cu)
target_link_libraries(blazingdb_ral_client ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(blazingdb_ral_service src/ral-service/ral-server.cu)
target_link_libraries(blazingdb_ral_service ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})

cuda_add_executable(ipc_server src/ipc_py.cu)
target_link_libraries(ipc_server ${LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})


