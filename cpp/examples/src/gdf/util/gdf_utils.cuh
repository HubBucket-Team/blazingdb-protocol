#pragma once 
#include <memory>
#include "bits.cuh"

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>
#include <cuda_runtime.h>

#include "gdf/gdf.h"
#include "../container/gdf_vector.cuh"

namespace gdf {
namespace util {


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


static std::basic_string<int8_t> BuildCudaIpcMemHandler (void *data) {
  cudaIpcMemHandle_t ipc_memhandle;
  cudaIpcGetMemHandle( &ipc_memhandle, (void*)data );
  cudaCheckErrors("Build IPC handle fail"); 

  std::basic_string<int8_t> bytes;
  bytes.resize(sizeof(cudaIpcMemHandle_t));
  memcpy((void*)bytes.data(), (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));
  return bytes;
}

static void* CudaIpcMemHandlerFrom (const std::basic_string<int8_t>& handler) {
  void * response = nullptr;
  std::cout << "handler-content: " <<  handler.size() <<  std::endl;
  if (handler.size() == 64) {
    cudaIpcMemHandle_t ipc_memhandle;
    memcpy((int8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
    cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("From IPC handle fail");       
  }
  return response;
}


static void create_sample_gdf_column(::gdf::container::GdfVector &one) {
  char * input1;
  size_t num_values = 32;
  input1 = new char[num_values];
  for(int i = 0; i < num_values; i++){
    input1[i] = i;
  }
  one.create_gdf_column(gdf::GDF_INT8, num_values, (void *) input1, sizeof(int8_t));
  //@todo: smart pointer, really smart?
  gdf::container::GDFRefCounter::getInstance()->deregister_column(one.get_gdf_column());
}
  
// Type for a unique_ptr to a gdf_column with a custom deleter
// Custom deleter is defined at construction
using gdf_col_pointer = typename std::unique_ptr<gdf_column, 
                                                 std::function<void(gdf_column*)>>;

template <typename col_type>
void print_typed_column(col_type * col_data, 
                        gdf_valid_type * validity_mask, 
                        const size_t num_rows)
{

  std::vector<col_type> h_data(num_rows);
  cudaMemcpy(h_data.data(), col_data, num_rows * sizeof(col_type), cudaMemcpyDeviceToHost);


  const size_t num_masks = valid_size(num_rows);
  std::vector<gdf_valid_type> h_mask(num_masks);
  if(nullptr != validity_mask)
  {
    cudaMemcpy(h_mask.data(), validity_mask, num_masks * sizeof(gdf_valid_type), cudaMemcpyDeviceToHost);
  }
  std::cout << "column :\n";

  if (validity_mask == nullptr) {
    for(size_t i = 0; i < num_rows; ++i)
    {
      if (sizeof(col_type) == 1)
        std::cout << (int)h_data[i] << " ";
      else
        std::cout << h_data[i] << " ";
    }
  } else {
    for(size_t i = 0; i < num_rows; ++i)
      {
        // If the element is valid, print it's value
        if(true == get_bit(h_mask.data(), i))
        {
          if (sizeof(col_type) == 1)
            std::cout << (int)h_data[i] << " ";
          else
            std::cout << h_data[i] << " ";
        }
        // Otherwise, print an @ to represent a null value
        else
        {
          std::cout << "@" << " ";
        }
      }
  }
  std::cout << std::endl;
}

void print_gdf_column(gdf_column const * the_column)
{
  const size_t num_rows = the_column->size;

  const gdf_dtype gdf_col_type = the_column->dtype;
  switch(gdf_col_type)
  {
    case GDF_INT8:
      {
        using col_type = int8_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT16:
      {
        using col_type = int16_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT32:
      {
        using col_type = int32_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_INT64:
      {
        using col_type = int64_t;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT32:
      {
        using col_type = float;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    case GDF_FLOAT64:
      {
        using col_type = double;
        col_type * col_data = static_cast<col_type*>(the_column->data);
        print_typed_column<col_type>(col_data, the_column->valid, num_rows);
        break;
      }
    default:
      {
        std::cout << "Attempted to print unsupported type.\n";
      }
  }
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Creates a unique_ptr that wraps a gdf_column structure intialized with a host vector
 *
 * @Param host_vector The host vector whose data is used to initialize the gdf_column
 *
 * @Returns A unique_ptr wrapping the new gdf_column
 */
/* ----------------------------------------------------------------------------*/
template <typename col_type>
gdf_col_pointer create_gdf_column(std::vector<col_type> const & host_vector,
                                  std::vector<gdf_valid_type> const & valid_vector = std::vector<gdf_valid_type>())
{
  // Deduce the type and set the gdf_dtype accordingly
  gdf_dtype gdf_col_type;
  if(std::is_same<col_type,int8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,uint8_t>::value) gdf_col_type = GDF_INT8;
  else if(std::is_same<col_type,int16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,uint16_t>::value) gdf_col_type = GDF_INT16;
  else if(std::is_same<col_type,int32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,uint32_t>::value) gdf_col_type = GDF_INT32;
  else if(std::is_same<col_type,int64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,uint64_t>::value) gdf_col_type = GDF_INT64;
  else if(std::is_same<col_type,float>::value) gdf_col_type = GDF_FLOAT32;
  else if(std::is_same<col_type,double>::value) gdf_col_type = GDF_FLOAT64;

  // Create a new instance of a gdf_column with a custom deleter that will free
  // the associated device memory when it eventually goes out of scope
  auto deleter = [](gdf_column* col){
                                      col->size = 0; 
                                      if(nullptr != col->data){cudaFree(col->data);} 
                                      if(nullptr != col->valid){cudaFree(col->valid);}
                                    };
  gdf_col_pointer the_column{new gdf_column, deleter};

  // Allocate device storage for gdf_column and copy contents from host_vector
  cudaMalloc(&(the_column->data), host_vector.size() * sizeof(col_type));
  cudaMemcpy(the_column->data, host_vector.data(), host_vector.size() * sizeof(col_type), cudaMemcpyHostToDevice);


  // If a validity bitmask vector was passed in, allocate device storage 
  // and copy its contents from the host vector
  if(valid_vector.size() > 0)
  {
    cudaMalloc(&(the_column->valid), valid_vector.size() * sizeof(gdf_valid_type));
    cudaMemcpy(the_column->valid, valid_vector.data(), valid_vector.size() * sizeof(gdf_valid_type), cudaMemcpyHostToDevice);
  }
  else
  {
    the_column->valid = nullptr;
  }

  // Fill the gdf_column members
  the_column->size = host_vector.size();
  the_column->dtype = gdf_col_type;
  gdf_dtype_extra_info extra_info;
  extra_info.time_unit = TIME_UNIT_NONE;
  the_column->dtype_info = extra_info;

  return the_column;
}
 

void DtoToGdfColumn(const std::vector<::gdf_dto::gdf_column> &columns) {
  for (auto &column : columns) {
      gdf::gdf_column gdf_pointer{
              .data = ::gdf::util::CudaIpcMemHandlerFrom(column.data),
              .valid = (unsigned char*)::gdf::util::CudaIpcMemHandlerFrom(column.valid),
              .size = column.size,
              .dtype = (::gdf::gdf_dtype)column.dtype,
              .null_count = column.null_count,
              .dtype_info = ::gdf::gdf_dtype_extra_info {
                .time_unit = (::gdf::gdf_time_unit)column.dtype_info.time_unit,
              },
          };
      //@todo: replace this function
      ::gdf::util::print_gdf_column( &gdf_pointer );
    }
}

void ToBlazingFrame(const ::blazingdb::protocol::TableGroupDTO& tableGroup) {
  for (auto& table : tableGroup.tables) { 
      DtoToGdfColumn(table.columns);
  }
}

} //util
} //gdf