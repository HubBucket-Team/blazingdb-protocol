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
#include "../../all_generated.h"
#include "gdf_dto.h"

namespace blazingdb {
namespace protocol {

struct BlazingTableDTO {
  std::string name;
  std::vector<::gdf_dto::gdf_column> columns;
  std::vector<std::string> columnNames;
  std::vector<uint64_t> columnTokens;
  uint64_t resultToken;
};

struct TableGroupDTO {
  std::vector<BlazingTableDTO> tables;
  std::string name;
};

static flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, const std::basic_string<int8_t> &reserved) {
  return builder.CreateVector(reserved.data(), reserved.size());
}

static std::basic_string<int8_t> CudaIpcMemHandlerFrom (const gdf::cudaIpcMemHandle_t *handler) {
  auto vector_bytes = handler->reserved();
  return std::basic_string<int8_t>{vector_bytes->data(), vector_bytes->size()};
}

static flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildDirectCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<int8_t> * data) {
  return builder.CreateVector(data->data(), data->size());
}

static std::vector<::gdf_dto::gdf_column>  GdfColumnsFrom(const flatbuffers::Vector<flatbuffers::Offset<blazingdb::protocol::gdf::gdf_column_handler>> *rawColumns) {
  std::vector<::gdf_dto::gdf_column>  columns;
  for (const auto& c : *rawColumns){
    bool valid_valid = (c->valid()->reserved()->size() == 64);
    bool custrings_membuffer_valid = (c->custrings_membuffer()->reserved()->size() == 64);
    bool custrings_views_valid = (c->custrings_views()->reserved()->size() == 64);
    ::gdf_dto::gdf_column column = {
        .data = CudaIpcMemHandlerFrom(c->data()),
        .valid = valid_valid ? CudaIpcMemHandlerFrom(c->valid()) : std::basic_string<int8_t>{},
        .size = c->size(),
        .dtype = (gdf_dto::gdf_dtype)c->dtype(),
        .null_count = c->null_count(),
        .dtype_info = gdf_dto::gdf_dtype_extra_info {
            .time_unit = (gdf_dto::gdf_time_unit) c->dtype_info()->time_unit(),
        },
        .custrings_views = custrings_views_valid ? CudaIpcMemHandlerFrom(c->custrings_views()) : std::basic_string<int8_t>{},
        .custrings_viewscount = c->custrings_viewscount(),
        .custrings_membuffer = custrings_membuffer_valid ? CudaIpcMemHandlerFrom(c->custrings_membuffer()) : std::basic_string<int8_t>{},
        .custrings_membuffersize = c->custrings_membuffersize(),
        .custrings_baseptr = c->custrings_baseptr()
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

static std::vector<uint64_t> ColumnTokensFrom(const flatbuffers::Vector<uint64_t> *rawColumnTokens) {
  std::vector<uint64_t> columnTokens;
  for (const auto& rawColumnToken : *rawColumnTokens){
    auto columnToken = rawColumnToken;
    columnTokens.push_back(columnToken);
  }
  return columnTokens;
}

static TableGroupDTO TableGroupDTOFrom(const blazingdb::protocol::TableGroup * tableGroup) {
  std::string name = std::string{tableGroup->name()->c_str()};
  std::vector<BlazingTableDTO> tables;

  auto rawTables = tableGroup->tables();
  for (const auto& table : *rawTables) {
    auto  columns = GdfColumnsFrom(table->columns());
    auto  columnNames = ColumnNamesFrom(table->columnNames());
    auto  columnTokens = ColumnTokensFrom(table->columnTokens());
    tables.push_back(BlazingTableDTO{
        .name = std::string{table->name()->c_str()},
        .columns = columns,
        .columnNames = columnNames,
        .columnTokens = columnTokens,
        .resultToken = table->resultToken()
    });
  }

  return TableGroupDTO {
      .tables = tables,
      .name = name,
  };
} 

std::vector<flatbuffers::Offset<gdf::gdf_column_handler>>  BuildFlatColumns(flatbuffers::FlatBufferBuilder &builder, const std::vector<::gdf_dto::gdf_column> &columns) {
  std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> offsets;
  for (auto & c: columns) {
    auto custrings_membuffer_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler(builder, c.custrings_membuffer) );
    auto custrings_views_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler(builder, c.custrings_views) );
    auto dtype_extra_info = gdf::Creategdf_dtype_extra_info (builder, (gdf::gdf_time_unit)c.dtype_info.time_unit );
    auto data_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler (builder, c.data) );
    auto valid_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler(builder, c.valid) );
    auto column_offset = ::blazingdb::protocol::gdf::Creategdf_column_handler(builder, data_offset, valid_offset, c.size, (gdf::gdf_dtype)c.dtype, dtype_extra_info, c.null_count, custrings_views_offset, c.custrings_viewscount, custrings_membuffer_offset, c.custrings_membuffersize, c.custrings_baseptr );
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

flatbuffers::Offset<flatbuffers::Vector<uint64_t>>  BuildFlatColumnTokens(flatbuffers::FlatBufferBuilder &builder, const std::vector<uint64_t> &columnTokens) {
  return builder.CreateVector(columnTokens.data(), columnTokens.size());
}

std::vector<flatbuffers::Offset<gdf::gdf_column_handler>>  BuildDirectFlatColumns(flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<flatbuffers::Offset<blazingdb::protocol::gdf::gdf_column_handler>> *rawColumns) {
  std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> offsets;
  for (const auto & c: *rawColumns) {
    auto custrings_views_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildDirectCudaIpcMemHandler(builder, c->custrings_views()->reserved()) );
    auto custrings_membuffer_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildDirectCudaIpcMemHandler(builder, c->custrings_membuffer()->reserved()) );
    auto dtype_extra_info = gdf::Creategdf_dtype_extra_info (builder, (gdf::gdf_time_unit)c->dtype_info()->time_unit() );
    auto data_offset =  gdf::CreatecudaIpcMemHandle_t(builder, BuildDirectCudaIpcMemHandler(builder, c->data()->reserved()) );
    auto valid_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildDirectCudaIpcMemHandler(builder, c->valid()->reserved()) );
    auto column_offset = ::blazingdb::protocol::gdf::Creategdf_column_handler(builder, data_offset, valid_offset, c->size(), (gdf::gdf_dtype)c->dtype(), dtype_extra_info, c->null_count(), custrings_views_offset, c->custrings_viewscount(), custrings_membuffer_offset, c->custrings_membuffersize(), c->custrings_baseptr() );
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

flatbuffers::Offset<flatbuffers::Vector<uint64_t>>  BuildDirectFlatColumnTokens(flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<uint64_t> *rawTokens) {
  std::vector<uint64_t> values;
  for (const auto & token: *rawTokens) {
    values.push_back( token );
  }
  return builder.CreateVector(values.data(), values.size());
}

static flatbuffers::Offset<TableGroup> BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                       const TableGroupDTO &tableGroup) {
  auto tableNameOffset = builder.CreateString(tableGroup.name);
  std::vector<flatbuffers::Offset<BlazingTable>> tablesOffset;

  for (auto table : tableGroup.tables) {
    auto columns = BuildFlatColumns(builder, table.columns);
    auto columnNames = BuildFlatColumnNames(builder, table.columnNames);
    auto token_offsets = BuildFlatColumnTokens(builder, table.columnTokens);
    tablesOffset.push_back( CreateBlazingTable(builder, builder.CreateString(table.name), builder.CreateVector(columns), builder.CreateVector(columnNames), token_offsets, table.resultToken));
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
    auto columnTokens = BuildDirectFlatColumnTokens(builder, table->columnTokens());
    tablesOffset.push_back( CreateBlazingTable(builder, 
                            builder.CreateString(table->name()->c_str()),
                            builder.CreateVector(columns), 
                            builder.CreateVector(columnNames),
                            columnTokens,
                            table->resultToken()));
  }

  auto tables = builder.CreateVector(tablesOffset);
  return CreateTableGroup(builder, tables, tableNameOffset);
}

}
}
#endif //BLAZINGDB_PROTOCOL_DTO_CUH_H
