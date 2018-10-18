#include <iostream>

#include <blazingdb/protocol/api.h>
#include <cuda_runtime.h>

#include "gdf/gdf.h"
#include "gdf/container/gdf_vector.cuh"
#include "gdf/util/gdf_utils.cuh"

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


static void* _CudaIpcMemHandlerFrom (const uint8_t *bytes) {
  void * response = nullptr;
  cudaIpcMemHandle_t ipc_memhandle;

  memcpy((int8_t*)&ipc_memhandle, bytes, sizeof(ipc_memhandle));
  cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
  cudaCheckErrors("IPC handle fail");

  return response;
}

int main() {
   blazingdb::protocol::UnixSocketConnection connection(
       {"/tmp/demo.socket", std::allocator<char>()});
   
   blazingdb::protocol::Server server(connection);
   server.handle([](const blazingdb::protocol::Buffer &buffer)
                     -> blazingdb::protocol::Buffer {
     
     std::cout << buffer.data() << std::endl;
    
    void *pointer = _CudaIpcMemHandlerFrom(buffer.data());
    gdf::gdf_column column {
        .data = pointer,                       /**< Pointer to the columns data */
        .valid = nullptr,            /**< Pointer to the columns validity bit mask where the 'i'th bit indicates if the 'i'th row is NULL */
        .size = 10,               /**< Number of data elements in the columns data buffer*/
        .dtype = gdf::gdf_dtype::GDF_INT32,                  /**< The datatype of the column's data */
        .null_count = 0,         /**< The number of NULL values in the column's data */
        .dtype_info = gdf::gdf_dtype_extra_info{
            .time_unit = (gdf::gdf_time_unit)0,
        }
    };
    gdf::util::print_gdf_column(&column);
    return blazingdb::protocol::Buffer(
         reinterpret_cast<const std::uint8_t *>("BlazingDB Response"), 18);
   });

  return 0;
}
