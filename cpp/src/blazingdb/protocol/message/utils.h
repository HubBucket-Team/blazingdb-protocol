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
#include "blazingdb/protocol/all_generated.h"
#include "interpreter/gdf_dto.h"

namespace blazingdb {
namespace protocol {

struct BlazingTableDTO {
  std::vector<::gdf_dto::gdf_column> columns;
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
    auto  columnTokens = ColumnTokensFrom(table->columnTokens());
    tables.push_back(BlazingTableDTO{
        .columns = columns,
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
    auto token_offsets = BuildFlatColumnTokens(builder, table.columnTokens);
    tablesOffset.push_back( CreateBlazingTable(builder, builder.CreateVector(columns), token_offsets, table.resultToken));
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
    auto columnTokens = BuildDirectFlatColumnTokens(builder, table->columnTokens());
    tablesOffset.push_back( CreateBlazingTable(builder, 
                            builder.CreateVector(columns), 
                            columnTokens,
                            table->resultToken()));
  }

  auto tables = builder.CreateVector(tablesOffset);
  return CreateTableGroup(builder, tables, tableNameOffset);
}

struct BlazingTableSchema {
  std::vector<::gdf_dto::gdf_column> columns;
  std::vector<uint64_t> columnTokens;
  uint64_t resultToken;

  static flatbuffers::Offset<blazingdb::protocol::BlazingTable> Serialize(flatbuffers::FlatBufferBuilder &builder, const BlazingTableSchema &data) {
    auto columnsOffset = BuildFlatColumns(builder, data.columns);
    auto columnTokensOffset = BuildFlatColumnTokens(builder, data.columnTokens);
    return blazingdb::protocol::CreateBlazingTable(builder, builder.CreateVector(columnsOffset), columnTokensOffset, data.resultToken);
  }

  static void Deserialize (const blazingdb::protocol::BlazingTable *pointer, BlazingTableSchema* schema){
      schema->columns = GdfColumnsFrom(pointer->columns());
      
      schema->columnTokens.clear();
      auto tokens_list = pointer->columnTokens();
      for (const auto &item : (*tokens_list)) {
        schema->columnTokens.push_back(item);
      }

      schema->resultToken = pointer->resultToken();
  }
};

struct TableSchemaSTL {
	std::vector<std::string> names;
	std::vector<uint64_t> calciteToFileIndices;
	std::vector<int> types;
	std::vector<uint64_t> numRowGroups;
  std::vector<std::string> files;
  std::string csvDelimiter;
  std::string csvLineTerminator;
  uint32_t csvSkipRows;

  static flatbuffers::Offset<blazingdb::protocol::TableSchema> Serialize(flatbuffers::FlatBufferBuilder &builder, const TableSchemaSTL &data) {
    auto namesOffset = builder.CreateVectorOfStrings(data.names);
    auto calciteToFileIndicesOffset = builder.CreateVector(data.calciteToFileIndices);
    auto typesOffset = builder.CreateVector(data.types);
    auto numRowGroupsOffset = builder.CreateVector(data.numRowGroups);
    auto filesOffset = builder.CreateVectorOfStrings(data.files);
    auto csvDelimiterOffset = builder.CreateString(data.csvDelimiter);
    auto csvLineTerminatorOffset = builder.CreateString(data.csvLineTerminator);

    return blazingdb::protocol::CreateTableSchema(builder, namesOffset, calciteToFileIndicesOffset, typesOffset, numRowGroupsOffset, filesOffset, csvDelimiterOffset, csvLineTerminatorOffset, data.csvSkipRows);
  }
  static void Deserialize (const blazingdb::protocol::TableSchema *pointer, TableSchemaSTL* schema){
      schema->names.clear();
      auto names_list = pointer->names();
      for (const auto &item : (*names_list)) {
        schema->names.push_back(std::string{item->c_str()});
      }

      schema->calciteToFileIndices.clear();
      auto calciteToFileIndices_list = pointer->calciteToFileIndices();
      for (const auto &item : (*calciteToFileIndices_list)) {
        schema->calciteToFileIndices.push_back(item);
      }

      schema->types.clear();
      auto types_list = pointer->types();
      for (const auto &item : (*types_list)) {
        schema->types.push_back(item);
      }

      schema->numRowGroups.clear();
      auto numRowGroups_list = pointer->numRowGroups();
      for (const auto &item : (*numRowGroups_list)) {
        schema->numRowGroups.push_back(item);
      }

      schema->files.clear();
      auto files_list = pointer->files();
      for (const auto &item : (*files_list)) {
        schema->files.push_back(std::string{item->c_str()});
      }

      schema->csvDelimiter = std::string{pointer->csvDelimiter()->c_str()};

      schema->csvLineTerminator = std::string{pointer->csvLineTerminator()->c_str()};

      schema->csvSkipRows = pointer->csvSkipRows();
  }
};

}
}
#endif //BLAZINGDB_PROTOCOL_DTO_CUH_H
