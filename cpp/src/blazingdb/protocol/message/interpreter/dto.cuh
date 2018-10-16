//
// Created by aocsa on 10/15/18.
//

#ifndef BLAZINGDB_PROTOCOL_DTO_CUH_H
#define BLAZINGDB_PROTOCOL_DTO_CUH_H

#include <string>
#include <functional>
#include <typeinfo>

#include <blazingdb/protocol/api.h>
#include <iostream>
#include "flatbuffers/flatbuffers.h"
#include "../generated/all_generated.h"
#include <cuda_runtime.h>

#include "libgdf.cuh"

namespace blazingdb {
namespace protocol {

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


struct BlazingTableDTO {
  std::string name;
  std::vector<::libgdf::gdf_column> columns;
  std::vector<std::string> columnNames;
};

struct TableGroupDTO {
  std::vector<BlazingTableDTO> tables;
  std::string name;
};


static flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, void *data) {

  cudaIpcMemHandle_t ipc_memhandle;
  cudaIpcGetMemHandle( &ipc_memhandle, (void*)data );
  cudaCheckErrors("Build IPC handle fail");

  int8_t* bytes = new int8_t[sizeof(cudaIpcMemHandle_t)];
  memcpy(bytes, (int8_t*)(&ipc_memhandle), sizeof(cudaIpcMemHandle_t));

  return builder.CreateVector(bytes, sizeof(cudaIpcMemHandle_t));
}

static flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildDirectCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<int8_t> * data) {
  return builder.CreateVector(data->data(), sizeof(cudaIpcMemHandle_t));
}


static void* CudaIpcMemHandlerFrom (const gdf::cudaIpcMemHandle_t *handler) {
  void * response = nullptr;
  cudaIpcMemHandle_t ipc_memhandle;

  auto bytes = handler->reserved();
  memcpy((int8_t*)&ipc_memhandle, bytes->data(), sizeof(ipc_memhandle));
  cudaIpcOpenMemHandle((void **)&response, ipc_memhandle, cudaIpcMemLazyEnablePeerAccess);
  cudaCheckErrors("From IPC handle fail");

  return response;
}

static std::vector<::libgdf::gdf_column>  GdfColumnsFrom(const flatbuffers::Vector<flatbuffers::Offset<blazingdb::protocol::gdf::gdf_column_handler>> *rawColumns) {
  std::vector<::libgdf::gdf_column>  columns;
  for (const auto& c : *rawColumns){
    ::libgdf::gdf_column column = {
        .data = CudaIpcMemHandlerFrom(c->data()),
        .valid = (unsigned char *)CudaIpcMemHandlerFrom(c->valid()),
        .size = c->size(),
        .dtype = (libgdf::gdf_dtype)c->dtype(),
        .null_count = c->null_count(),
        .dtype_info = libgdf::gdf_dtype_extra_info {
            .time_unit = (libgdf::gdf_time_unit) c->dtype_info()->time_unit()
        },
    };
    columns.push_back(column);
  }
  return columns;
}

static std::vector<std::string> ColumnNamesFrom(const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *rawNames) {
  std::vector<std::string> columnNames;
  for (const auto& rawName : *rawNames){
    auto name = std::string{rawName->c_str()};  
    columnNames.push_back(name);
  }
  return columnNames;
}

static TableGroupDTO TableGroupDTOFrom(const blazingdb::protocol::TableGroup * tableGroup) {
  std::string name = std::string{tableGroup->name()->c_str()};
  std::vector<BlazingTableDTO> tables;

  auto rawTables = tableGroup->tables();
  for (const auto& table : *rawTables) {
    auto  columns = GdfColumnsFrom(table->columns());
    auto  columnNames = ColumnNamesFrom(table->columnNames());
    tables.push_back(BlazingTableDTO{
        .name = std::string{table->name()->c_str()},
        .columns = columns,
        .columnNames = columnNames,
    });
  }

  return TableGroupDTO {
      .tables = tables,
      .name = name,
  };
}

std::vector<flatbuffers::Offset<gdf::gdf_column_handler>>  BuildFlatColumns(flatbuffers::FlatBufferBuilder &builder, const std::vector<::libgdf::gdf_column> &columns) {
  std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> offsets;
  for (auto & c: columns) {
    auto dtype_extra_info = gdf::Creategdf_dtype_extra_info (builder, (gdf::gdf_time_unit)c.dtype_info.time_unit );
    auto data_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler (builder, c.data) );
    auto valid_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler(builder, c.valid) );
    auto column_offset = ::blazingdb::protocol::gdf::Creategdf_column_handler(builder, data_offset, valid_offset, c.size, (gdf::gdf_dtype)c.dtype, dtype_extra_info);
    offsets.push_back(column_offset);
  }
  return offsets;
};

std::vector<flatbuffers::Offset<flatbuffers::String>>  BuildFlatColumnNames(flatbuffers::FlatBufferBuilder &builder, const std::vector<std::string> &columnNames) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> offsets;
  for (auto & name: columnNames) {
    offsets.push_back( builder.CreateString(name.data()));
  }
  return offsets;
};


std::vector<flatbuffers::Offset<gdf::gdf_column_handler>>  BuildDirectFlatColumns(flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<flatbuffers::Offset<blazingdb::protocol::gdf::gdf_column_handler>> *rawColumns) {
  std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> offsets;
  for (const auto & c: *rawColumns) {
    auto dtype_extra_info = gdf::Creategdf_dtype_extra_info (builder, (gdf::gdf_time_unit)c->dtype_info()->time_unit() );
     auto data_offset =  gdf::CreatecudaIpcMemHandle_t(builder, BuildDirectCudaIpcMemHandler(builder, c->data()->reserved()) );
     auto valid_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildDirectCudaIpcMemHandler(builder, c->valid()->reserved()) );
    auto column_offset = ::blazingdb::protocol::gdf::Creategdf_column_handler(builder, data_offset, valid_offset, c->size(), (gdf::gdf_dtype)c->dtype(), dtype_extra_info);
    offsets.push_back(column_offset);
  }
  return offsets;
};

std::vector<flatbuffers::Offset<flatbuffers::String>>  BuildDirectFlatColumnNames(flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *rawNames) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> offsets;
  for (const auto & name: *rawNames) {
    offsets.push_back( builder.CreateString(name->c_str()));
  }
  return offsets;
};

static flatbuffers::Offset<TableGroup> BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                       const TableGroupDTO &tableGroup) {
  auto tableNameOffset = builder.CreateString(tableGroup.name);
  std::vector<flatbuffers::Offset<BlazingTable>> tablesOffset;

  for (auto table : tableGroup.tables) {
    auto columns = BuildFlatColumns(builder, table.columns);
    auto columnNames = BuildFlatColumnNames(builder, table.columnNames);
    tablesOffset.push_back( CreateBlazingTable(builder, builder.CreateString(table.name), builder.CreateVector(columns), builder.CreateVector(columnNames)));
  }

  auto tables = builder.CreateVector(tablesOffset);
  return CreateTableGroup(builder, tables, tableNameOffset);
}


static flatbuffers::Offset<TableGroup> BuildDirectTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                       const blazingdb::protocol::TableGroup *tableGroup) { 
  auto tableNameOffset = builder.CreateString(tableGroup->name()->c_str());
  std::vector<flatbuffers::Offset<BlazingTable>> tablesOffset;
  auto rawTables = tableGroup->tables();
  for (const auto &table : *rawTables) {
    auto columns = BuildDirectFlatColumns(builder, table->columns());
    auto columnNames = BuildDirectFlatColumnNames(builder, table->columnNames());
    tablesOffset.push_back( CreateBlazingTable(builder, 
                            builder.CreateString(table->name()->c_str()),
                            builder.CreateVector(columns), 
                            builder.CreateVector(columnNames)));
  }

  auto tables = builder.CreateVector(tablesOffset);
  return CreateTableGroup(builder, tables, tableNameOffset);
}

}
}
#endif //BLAZINGDB_PROTOCOL_DTO_CUH_H
