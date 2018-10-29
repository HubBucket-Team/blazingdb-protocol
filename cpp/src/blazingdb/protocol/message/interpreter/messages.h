#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"
#include "utils.h"
#include "gdf_dto.h"

namespace blazingdb {
namespace protocol {

namespace interpreter {


class ExecutePlanDirectRequestMessage  : public IMessage {
public:

  ExecutePlanDirectRequestMessage(const std::string &logicalPlan, const blazingdb::protocol::TableGroup *tableGroup)
      : IMessage(), logicalPlan{logicalPlan}, tableGroup{tableGroup}
  {

  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder;
    auto logicalPlan_offset = builder.CreateString(logicalPlan);
    auto tableGroupOffset = ::blazingdb::protocol::BuildDirectTableGroup(builder, tableGroup);
    builder.Finish(interpreter::CreateDMLRequest(builder, logicalPlan_offset, tableGroupOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

private:
  std::string logicalPlan;
  const blazingdb::protocol::TableGroup * tableGroup;
};

class ExecutePlanRequestMessage  : public IMessage {
public:

  ExecutePlanRequestMessage(const std::string &logicalPlan, const  ::blazingdb::protocol::TableGroupDTO &tableGroup)
      : IMessage(), logicalPlan{logicalPlan}, tableGroup{tableGroup}
  {

  }
  ExecutePlanRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::DMLRequest>(buffer);
    logicalPlan = std::string{pointer->logicalPlan()->c_str()};
    std::cout << "query-stirng-log>>" << logicalPlan << std::endl;
    tableGroup =  ::blazingdb::protocol::TableGroupDTOFrom(pointer->tableGroup());
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder;
    auto logicalPlan_offset = builder.CreateString(logicalPlan);
    auto tableGroupOffset = ::blazingdb::protocol::BuildTableGroup(builder, tableGroup);
    builder.Finish(interpreter::CreateDMLRequest(builder, logicalPlan_offset, tableGroupOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getLogicalPlan() {
    return logicalPlan;
  }

  ::blazingdb::protocol::TableGroupDTO getTableGroup() {
    return tableGroup;
  }

private:
  std::string logicalPlan;
  ::blazingdb::protocol::TableGroupDTO tableGroup;
};




class  GetResultRequestMessage :  public TypedMessage<uint64_t, interpreter::GetResultRequest> {
  public:

  GetResultRequestMessage(const std::uint64_t& value)
      : TypedMessage<uint64_t, interpreter::GetResultRequest>(value)
  {
  }

  GetResultRequestMessage (const uint8_t* buffer)
      :  TypedMessage<uint64_t, interpreter::GetResultRequest>(buffer, &interpreter::GetResultRequest::resultToken)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(interpreter::CreateGetResultRequest);
  }

  uint64_t  getResultToken () {
    return value_;
  }
};


struct NodeConnectionDTO {
  std::string path;
  NodeConnectionType type;
};

class  ExecutePlanResponseMessage : public IMessage {
public:
  ExecutePlanResponseMessage (uint64_t resultToken, const NodeConnectionDTO &nodeInfo)
    : IMessage(), resultToken{resultToken}, nodeInfo{nodeInfo}
  {

  }

  ExecutePlanResponseMessage (const uint8_t* buffer)
      :   IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::ExecutePlanResponse>(buffer);
    resultToken = pointer->resultToken();
    nodeInfo = NodeConnectionDTO {
      .path = std::string{pointer->nodeConnection()->path()->c_str()},
      .type = pointer->nodeConnection()->type()
    };
  };
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto nodeInfo_offset = CreateNodeConnectionDirect(builder, nodeInfo.path.data(), nodeInfo.type);
    auto root = interpreter::CreateExecutePlanResponse(builder, resultToken, nodeInfo_offset);
    builder.Finish(root);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  uint64_t getResultToken() {
    return resultToken;
  }

  NodeConnectionDTO getNodeInfo() {
    return nodeInfo;
  }
public:
  uint64_t resultToken;
  NodeConnectionDTO nodeInfo;
};


struct BlazingMetadataDTO{
  std::string status;
  std::string message;
  double time;
  int rows;
};


class GetResultResponseMessage : public IMessage {
public:
  GetResultResponseMessage (const BlazingMetadataDTO&  metadata, const std::vector<std::string>& columnNames, const std::vector<::gdf_dto::gdf_column>& columns)
      : IMessage(), metadata{metadata}, columnNames{columnNames}, columns{columns}
  {

  }

  GetResultResponseMessage (const uint8_t* buffer)
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
    columns = GdfColumnsFrom(pointer->columns());
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto metadata_offset = blazingdb::protocol::interpreter::CreateBlazingMetadataDirect(builder, metadata.status.data(), metadata.message.data(), metadata.time, metadata.rows);

    auto names_offset = BuildFlatColumnNames(builder, columnNames);
    auto values_offset = BuildFlatColumns(builder, columns);

    auto root = interpreter::CreateGetResultResponse(builder, metadata_offset, builder.CreateVector(values_offset), builder.CreateVector(names_offset));
    builder.Finish(root);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  BlazingMetadataDTO getMetadata()
  {
    return metadata;
  }

  std::vector<std::string> getColumnNames()
  {
    return columnNames;
  }

  std::vector<::gdf_dto::gdf_column> getColumns()
  {
    return columns;
  }

public:
  BlazingMetadataDTO  metadata;
  std::vector<std::string> columnNames;
  std::vector<::gdf_dto::gdf_column> columns;
};


} // interpreter
} // protocol
} // blazingdb
