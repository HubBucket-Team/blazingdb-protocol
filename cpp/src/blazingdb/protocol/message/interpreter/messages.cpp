#include "blazingdb/protocol/message/interpreter/messages.h"

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"
#include "utils.h"
#include "gdf_dto.h"

namespace blazingdb {
namespace protocol {

namespace interpreter {


  ExecutePlanDirectRequestMessage::ExecutePlanDirectRequestMessage(const std::string &logicalPlan, const blazingdb::protocol::TableGroup *tableGroup)
      : IMessage(), logicalPlan{logicalPlan}, tableGroup{tableGroup}
  {

  }

  std::shared_ptr<flatbuffers::DetachedBuffer> ExecutePlanDirectRequestMessage::getBufferData( ) const   {
    flatbuffers::FlatBufferBuilder builder;
    auto logicalPlan_offset = builder.CreateString(logicalPlan);
    auto tableGroupOffset = ::blazingdb::protocol::BuildDirectTableGroup(builder, tableGroup);
    builder.Finish(interpreter::CreateDMLRequest(builder, logicalPlan_offset, tableGroupOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }


  ExecutePlanRequestMessage::ExecutePlanRequestMessage(const std::string &logicalPlan, const  ::blazingdb::protocol::TableGroupDTO &tableGroup)
      : IMessage(), logicalPlan{logicalPlan}, tableGroup{tableGroup}
  {

  }
  ExecutePlanRequestMessage::ExecutePlanRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::DMLRequest>(buffer);
    logicalPlan = std::string{pointer->logicalPlan()->c_str()};
    std::cout << "query-stirng-log>>" << logicalPlan << std::endl;
    tableGroup =  ::blazingdb::protocol::TableGroupDTOFrom(pointer->tableGroup());
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> ExecutePlanRequestMessage::getBufferData( ) const   {
    flatbuffers::FlatBufferBuilder builder;
    auto logicalPlan_offset = builder.CreateString(logicalPlan);
    auto tableGroupOffset = ::blazingdb::protocol::BuildTableGroup(builder, tableGroup);
    builder.Finish(interpreter::CreateDMLRequest(builder, logicalPlan_offset, tableGroupOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string ExecutePlanRequestMessage::getLogicalPlan() {
    return logicalPlan;
  }

  ::blazingdb::protocol::TableGroupDTO ExecutePlanRequestMessage::getTableGroup() {
    return tableGroup;
  }

  



/*
  GetResultRequestMessage::GetResultRequestMessage(const std::uint64_t& value)
      : TypedMessage<uint64_t, interpreter::GetResultRequest>(value)
  {
  }

  GetResultRequestMessage::GetResultRequestMessage (const uint8_t* buffer)
      :  TypedMessage<uint64_t, interpreter::GetResultRequest>(buffer, &interpreter::GetResultRequest::resultToken)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> GetResultRequestMessage::getBufferData( ) const   {
    return this->getBufferDataUsing(interpreter::CreateGetResultRequest);
  }

  uint64_t  GetResultRequestMessage::getResultToken () {
    return value_;
  }
*/

  ExecutePlanResponseMessage::ExecutePlanResponseMessage (uint64_t resultToken, const NodeConnectionDTO &nodeInfo)
    : IMessage(), resultToken{resultToken}, nodeInfo{nodeInfo}
  {

  }

  ExecutePlanResponseMessage::ExecutePlanResponseMessage (const uint8_t* buffer)
      :   IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::ExecutePlanResponse>(buffer);
    resultToken = pointer->resultToken();
    nodeInfo = NodeConnectionDTO {
      .port = pointer->nodeConnection()->port(),
      .path = std::string{pointer->nodeConnection()->path()->c_str()},
      .type = pointer->nodeConnection()->type()
    };
  };
  std::shared_ptr<flatbuffers::DetachedBuffer> ExecutePlanResponseMessage::getBufferData( ) const   {
    flatbuffers::FlatBufferBuilder builder{0};
    auto nodeInfo_offset = CreateNodeConnectionDirect(builder, nodeInfo.port, nodeInfo.path.data(), nodeInfo.type);
    auto root = interpreter::CreateExecutePlanResponse(builder, resultToken, nodeInfo_offset);
    builder.Finish(root);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  uint64_t ExecutePlanResponseMessage::getResultToken() {
    return resultToken;
  }

  NodeConnectionDTO ExecutePlanResponseMessage::getNodeInfo() {
    return nodeInfo;
  }

  


  GetResultResponseMessage::GetResultResponseMessage (const BlazingMetadataDTO&  metadata, const std::vector<std::string>& columnNames, const std::vector<uint64_t>& columnTokens, const std::vector<::gdf_dto::gdf_column>& columns)
      : IMessage(), metadata{metadata}, columnNames{columnNames}, columnTokens{columnTokens}, columns{columns}
  {

  }

  GetResultResponseMessage::GetResultResponseMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::GetResultResponse>(buffer);
    metadata = BlazingMetadataDTO {
        .status = std::string{pointer->metadata()->status()->c_str()},
        .message = std::string{pointer->metadata()->message()->c_str()},
        .time = pointer->metadata()->time(),
        .rows = pointer->metadata()->rows()
    };

    columnNames = ColumnNamesFrom(pointer->columnNames());
    columnTokens = ColumnTokensFrom(pointer->columnTokens());
    columns = GdfColumnsFrom(pointer->columns());
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> GetResultResponseMessage::getBufferData( ) const   {
    flatbuffers::FlatBufferBuilder builder{0};
    auto metadata_offset = blazingdb::protocol::interpreter::CreateBlazingMetadataDirect(builder, metadata.status.data(), metadata.message.data(), metadata.time, metadata.rows);

    auto names_offset = BuildFlatColumnNames(builder, columnNames);
    auto values_offset = BuildFlatColumns(builder, columns);
    auto tokens_offset = BuildFlatColumnTokens(builder, columnTokens);

    auto root = interpreter::CreateGetResultResponse(builder, metadata_offset, builder.CreateVector(values_offset), builder.CreateVector(names_offset), tokens_offset);
    builder.Finish(root);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  BlazingMetadataDTO GetResultResponseMessage::getMetadata()
  {
    return metadata;
  }

  std::vector<std::string> GetResultResponseMessage::getColumnNames()
  {
    return columnNames;
  }

  std::vector<uint64_t> GetResultResponseMessage::getColumnTokens()
  {
    return columnTokens;
  }

  std::vector<::gdf_dto::gdf_column> GetResultResponseMessage::getColumns()
  {
    return columns;
  }

  


  CreateTableResponseMessage::CreateTableResponseMessage (const blazingdb::protocol::TableSchemaSTL& tableSchema)
    : IMessage(), tableSchema{tableSchema}
  {

  }

  CreateTableResponseMessage::CreateTableResponseMessage (const uint8_t* buffer)
      :   IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::orchestrator::DDLCreateTableResponse>(buffer);
    blazingdb::protocol::TableSchemaSTL::Deserialize(pointer->tableSchema(), &tableSchema);
  };

  std::shared_ptr<flatbuffers::DetachedBuffer> CreateTableResponseMessage::getBufferData() const   {
    flatbuffers::FlatBufferBuilder builder{0};
    auto tableSchemaOffset = blazingdb::protocol::TableSchemaSTL::Serialize(builder, tableSchema);
    builder.Finish(orchestrator::CreateDDLCreateTableResponse(builder, tableSchemaOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  const blazingdb::protocol::TableSchemaSTL& CreateTableResponseMessage::getTableSchema() {
    return tableSchema;
  }


} // interpreter
} // protocol
} // blazingdb
