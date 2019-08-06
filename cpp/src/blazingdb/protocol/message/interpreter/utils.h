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
#include <blazingdb/protocol/all_generated.h>
#include "gdf_dto.h"

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

flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, const std::basic_string<int8_t> &reserved);

std::basic_string<int8_t> CudaIpcMemHandlerFrom (const gdf::cudaIpcMemHandle_t *handler);

flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildDirectCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, const flatbuffers::Vector<int8_t> * data);

flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildCustringsData (flatbuffers::FlatBufferBuilder &builder, const std::basic_string<int8_t> &reserved);

std::basic_string<int8_t> CustringsDataFrom (const gdf::custringsData_t *handler);

std::vector<::gdf_dto::gdf_column>  GdfColumnsFrom(const flatbuffers::Vector<flatbuffers::Offset<blazingdb::protocol::gdf::gdf_column_handler>> *rawColumns);

std::vector<std::string> ColumnNamesFrom(const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *rawNames);

std::vector<uint64_t> ColumnTokensFrom(const flatbuffers::Vector<uint64_t> *rawColumnTokens);

std::vector<flatbuffers::Offset<gdf::gdf_column_handler>>  BuildFlatColumns(flatbuffers::FlatBufferBuilder &builder, const std::vector<::gdf_dto::gdf_column> &columns);

std::vector<flatbuffers::Offset<flatbuffers::String>>  BuildFlatColumnNames(flatbuffers::FlatBufferBuilder &builder, const std::vector<std::string> &columnNames);

flatbuffers::Offset<flatbuffers::Vector<uint64_t>>  BuildFlatColumnTokens(flatbuffers::FlatBufferBuilder &builder, const std::vector<uint64_t> &columnTokens);

flatbuffers::Offset<TableGroup> BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                       const TableGroupDTO &tableGroup);


struct BlazingTableSchema {
  std::vector<::gdf_dto::gdf_column> columns;
  std::vector<uint64_t> columnTokens;
  uint64_t resultToken;

  static flatbuffers::Offset<blazingdb::protocol::BlazingTable> Serialize(flatbuffers::FlatBufferBuilder &builder, const BlazingTableSchema &data);

  static void Deserialize (const blazingdb::protocol::BlazingTable *pointer, BlazingTableSchema* schema);
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
  int32_t csvHeader;
  int32_t csvNrows;

  static flatbuffers::Offset<blazingdb::protocol::TableSchema> Serialize(flatbuffers::FlatBufferBuilder &builder, const TableSchemaSTL &data);
  static void Deserialize (const blazingdb::protocol::TableSchema *pointer, TableSchemaSTL* schema);
};

}
}
#endif //BLAZINGDB_PROTOCOL_DTO_CUH_H
