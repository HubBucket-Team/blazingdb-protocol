/*
 * GDFColumn.h
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#ifndef GDFCOLUMN_H_
#define GDFCOLUMN_H_

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>
#include "GDFCounter.cuh"
#include "libgdf.h"
#include <cuda_runtime.h>

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

namespace libgdf {

class gdf_column_cpp
{
	private:
		gdf_column column;

	public:

    void* data();

    gdf_valid_type* valid();

    gdf_size_type size();

    gdf_dtype dtype();

    gdf_size_type null_count();

    gdf_dtype_extra_info dtype_info();

	void set_dtype(gdf_dtype dtype);

	gdf_column_cpp();

	gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	gdf_column_cpp(const gdf_column_cpp& col);

	gdf_column_cpp(gdf_column_cpp& col);

	void operator=(const gdf_column_cpp& col);

	gdf_column* get_gdf_column();

	void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	void realloc_gdf_column(gdf_dtype type, size_t size, size_t width);

	gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype);

	~gdf_column_cpp();
};


gdf_column_cpp::gdf_column_cpp()
{
    column.data = nullptr;
    column.valid = nullptr;
    column.size = 0;
    column.dtype = GDF_invalid;
    column.null_count = 0;
}

gdf_column_cpp::gdf_column_cpp(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    create_gdf_column(type, num_values, input_data, width_per_value);
}

gdf_column_cpp::gdf_column_cpp(const gdf_column_cpp& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;

    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));
}

gdf_column_cpp::gdf_column_cpp(gdf_column_cpp& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;

    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));
}

void gdf_column_cpp::operator=(const gdf_column_cpp& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;

    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));
}

gdf_column* gdf_column_cpp::get_gdf_column()
{
    return &column;
}

void gdf_column_cpp::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    char * data;
    gdf_valid_type * valid_device;

    size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

    cudaMalloc((void **) &valid_device, allocation_size_valid);	

    cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid); //assume all relevant bits are set to on

    cudaMalloc((void **) &data, width_per_value * num_values);

    if(input_data != nullptr){
        cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
    }

    gdf_column_view(&this->column, (void *) data, valid_device, num_values, type);

    GDFRefCounter::getInstance()->register_column(&this->column);
}

void gdf_column_cpp::realloc_gdf_column(gdf_dtype type, size_t size, size_t width){
    GDFRefCounter::getInstance()->decrement(&this->column); //decremeting reference, deallocating space

	create_gdf_column(type, size, nullptr, width);
}

gdf_error gdf_column_cpp::gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype)
{
    column->data = data;
    column->valid = valid;
    column->size = size;
    column->dtype = dtype;
    column->null_count = 0;
    return GDF_SUCCESS;
}

gdf_column_cpp::~gdf_column_cpp()
{
    GDFRefCounter::getInstance()->decrement(&this->column);
}

void* gdf_column_cpp::data(){
    return column.data;
}

gdf_valid_type* gdf_column_cpp::valid(){
    return column.valid;
}
gdf_size_type gdf_column_cpp::size(){
    return column.size;
}

gdf_dtype gdf_column_cpp::dtype(){
    return column.dtype;
}

gdf_size_type gdf_column_cpp::null_count(){
    return column.null_count;
}

gdf_dtype_extra_info gdf_column_cpp::dtype_info(){
    return column.dtype_info;
}

void gdf_column_cpp::set_dtype(gdf_dtype dtype){
    column.dtype=dtype;
}



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
  cudaIpcMemHandle_t ipc_memhandle;

  memcpy((int8_t*)&ipc_memhandle, handler.data(), sizeof(ipc_memhandle));
  cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
  cudaCheckErrors("From IPC handle fail");
  return response;
}


#define GDF_VALID_BITSIZE 8

static void create_sample_gdf_column(libgdf::gdf_column_cpp &one) {
  char * input1;
  size_t num_values = 32;
  input1 = new char[num_values];
  for(int i = 0; i < num_values; i++){
    input1[i] = i;
  }
  one.create_gdf_column(libgdf::GDF_INT8, num_values, (void *) input1, 1);
  GDFRefCounter::getInstance()->deregister_column(one.get_gdf_column());
}

//todo: rehacer gdf-cpp- and utils
static void print_column(gdf_column * column){
	char * host_data_out = new char[column->size];
	char * host_valid_out;
	if(column->size % 8 != 0){
		host_valid_out = new char[(column->size + (8 - (column->size % 8)))/8];
	}else{
		host_valid_out = new char[column->size / 8];
	}
	cudaMemcpy(host_data_out,column->data,sizeof(int8_t) * column->size, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_valid_out,column->valid,sizeof(int8_t) * (column->size + GDF_VALID_BITSIZE - 1) / GDF_VALID_BITSIZE, cudaMemcpyDeviceToHost);
	std::cout<<"Printing Column address ptr: "<<column<<", Size: "<<column->size<<", Type: "<<column->dtype<<"\n"<<std::flush;
	for(int i = 0; i < column->size; i++){
		int col_position = i / 8;
		int bit_offset = 8 - (i % 8);
		std::cout<<"host_data_out["<<i<<"] = "<<((int)host_data_out[i])<<" valid="<<((host_valid_out[col_position] >> bit_offset ) & 1)<<std::endl;
	}

	delete[] host_data_out;
	delete[] host_valid_out;

	std::cout<<std::endl<<std::endl;
}


void DtoToGdfColumn(const std::vector<::gdf_dto::gdf_column> &columns) {
  for (auto &column : columns) {
      ::libgdf::gdf_column gdf_pointer{
              .data = libgdf::CudaIpcMemHandlerFrom(column.data),
              .valid = (unsigned char*)libgdf::CudaIpcMemHandlerFrom(column.valid),
              .size = column.size,
              .dtype = (libgdf::gdf_dtype)column.dtype,
              .null_count = column.null_count,
              .dtype_info = libgdf::gdf_dtype_extra_info {
                .time_unit = (libgdf::gdf_time_unit)column.dtype_info.time_unit,
              },
          };
      libgdf::print_column( &gdf_pointer );
    }
}

void ToBlazingFrame(const ::blazingdb::protocol::TableGroupDTO& tableGroup) {
  for (auto& table : tableGroup.tables) { 
      DtoToGdfColumn(table.columns);
  }
}


}
#endif /* GDFCOLUMN_H_ */