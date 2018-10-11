#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"

namespace blazingdb {
namespace protocol {

namespace interpreter {

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
    return this->getBufferDataUsing(orchestrator::CreateDMLResponse);
  }

  uint64_t  getResultToken () {
    return value_;
  }
};


struct NodeConnectionInformationDTO {
  std::string path;
  NodeConnectionType type;
};

class  ExecutePlanResponseMessage : public IMessage {
public:
  ExecutePlanResponseMessage (uint64_t resultToken, const NodeConnectionInformationDTO &nodeInfo)
    : IMessage(), resultToken{resultToken}, nodeInfo{nodeInfo}
  {

  }

  ExecutePlanResponseMessage (const uint8_t* buffer)
      :   IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::interpreter::ExecutePlanResponse>(buffer);
    resultToken = pointer->resultToken();
    nodeInfo = NodeConnectionInformationDTO {
      .path = std::string{pointer->connectionInfo()->path()->c_str()},
      .type = pointer->connectionInfo()->type()
    };
  };
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto nodeInfo_offset = CreateNodeConnectionInformationDirect(builder, nodeInfo.path.data(), nodeInfo.type);
    auto root = interpreter::CreateExecutePlanResponse(builder, resultToken, nodeInfo_offset);
    builder.Finish(root);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  uint64_t getResultToken() {
    return resultToken;
  }

  NodeConnectionInformationDTO getNodeInfo() {
    return nodeInfo;
  }
public:
  uint64_t resultToken;
  NodeConnectionInformationDTO nodeInfo;
};

} // interpreter
} // protocol
} // blazingdb
