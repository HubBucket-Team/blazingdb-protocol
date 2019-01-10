#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"

#include "../messages.h"
#include "../interpreter/messages.h"

namespace blazingdb {
namespace protocol {

namespace orchestrator {


class DMLRequestMessage  : public IMessage {
public:
  DMLRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::orchestrator::DMLRequest>(buffer);
    query = std::string{pointer->query()->c_str()};
    tableGroup =  pointer->tableGroup();
  }

  std::string getQuery () {
    return query;
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return nullptr;
  }

  const blazingdb::protocol::TableGroup * getTableGroup() {
    return tableGroup;
  }
private:
  std::string query;
  const blazingdb::protocol::TableGroup * tableGroup;
};

class DMLResponseMessage : public IMessage {
public:
  using NodeConnectionDTO = blazingdb::protocol::interpreter::NodeConnectionDTO;

  DMLResponseMessage(const std::uint64_t resultToken,
                     NodeConnectionDTO & nodeInfo,
                     const std::int64_t  calciteTime)
      : IMessage(), resultToken{resultToken}, nodeInfo{nodeInfo},
        calciteTime_{calciteTime} {}

  DMLResponseMessage(const uint8_t *buffer) : IMessage() {
    auto pointer =
        flatbuffers::GetRoot<blazingdb::protocol::orchestrator::DMLResponse>(
            buffer);
    resultToken = pointer->resultToken();
    nodeInfo    = NodeConnectionDTO{
        .port = pointer->nodeConnection()->port();
        .path = std::string{pointer->nodeConnection()->path()->c_str()},
        .type = pointer->nodeConnection()->type()};
    calciteTime_ = pointer->calciteTime();
  };

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final {
    flatbuffers::FlatBufferBuilder builder{0};
    auto                           nodeInfo_offset = CreateNodeConnectionDirect(
        builder, nodeInfo.path.data(), nodeInfo.type);
    auto root = orchestrator::CreateDMLResponse(
      builder, resultToken, nodeInfo_offset, calciteTime_);
    builder.Finish(root);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::uint64_t getResultToken() const noexcept { return resultToken; }

  NodeConnectionDTO getNodeInfo() const noexcept { return nodeInfo; }

  std::int64_t getCalciteTime() const noexcept { return calciteTime_; }

public:
  std::uint64_t     resultToken;
  NodeConnectionDTO nodeInfo;
  std::int64_t      calciteTime_;
};

class DDLRequestMessage : public StringTypeMessage<orchestrator::DDLRequest> {
public:
  DDLRequestMessage(const std::string& string_value)
      : StringTypeMessage<orchestrator::DDLRequest>(string_value)
  {
  }

  DDLRequestMessage (const uint8_t* buffer)
      :  StringTypeMessage<orchestrator::DDLRequest>(buffer, &orchestrator::DDLRequest::query)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDDLRequestDirect);
  }

  std::string getQuery () {
    return string_value;
  }
};


class DDLDropTableRequestMessage : public IMessage {
public:

  DDLDropTableRequestMessage()
      : IMessage()
  {

  }
  DDLDropTableRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::orchestrator::DDLDropTableRequest>(buffer);
    name = std::string{pointer->name()->c_str()};
    dbName = std::string{pointer->dbName()->c_str()};

  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder;
    auto name_offset = builder.CreateString(name);

    auto dbname_offset = builder.CreateString(dbName);

    builder.Finish(orchestrator::CreateDDLDropTableRequest(builder, name_offset, dbname_offset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
public:
  std::string name;
  std::string dbName;
};


class DDLCreateTableRequestMessage : public IMessage {
public:

  DDLCreateTableRequestMessage()
    : IMessage()
  {

  }
  DDLCreateTableRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::orchestrator::DDLCreateTableRequest>(buffer);
    name = std::string{pointer->name()->c_str()};
    dbName = std::string{pointer->dbName()->c_str()};
    auto name_list = pointer->columnNames();
    for (const auto &item : (*name_list)) {
      columnNames.push_back(std::string{item->c_str()});
    }

    auto type_list = pointer->columnTypes();
    for (const auto &item : (*type_list)) {
      columnTypes.push_back(std::string{item->c_str()});
    }
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder;
    auto name_offset = builder.CreateString(name);
    auto vectorOfColumnNames = builder.CreateVectorOfStrings(columnNames);
    auto vectorOfColumnTypes = builder.CreateVectorOfStrings(columnTypes);
    auto dbname_offset = builder.CreateString(dbName);
    builder.Finish(orchestrator::CreateDDLCreateTableRequest(builder, name_offset,vectorOfColumnNames, vectorOfColumnTypes, dbname_offset));

    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
public:
  std::string name;
  std::vector<std::string> columnNames;
  std::vector<std::string> columnTypes;
  std::string dbName;

};

// authorization


class AuthResponseMessage : public IMessage {
public:

  AuthResponseMessage(int64_t access_token)
      : IMessage{}, access_token_{access_token}
  {
  }

  AuthResponseMessage ( AuthResponseMessage && ) = default;

  AuthResponseMessage (const uint8_t* buffer)
      :   IMessage{}
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::orchestrator::AuthResponse>(buffer);

    access_token_ = pointer->accessToken();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto root_offset = CreateAuthResponse(builder, access_token_);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  int64_t getAccessToken () {
    return access_token_;
  }

private:
  int64_t access_token_;
};

} // orchestrator
} // protocol
} // blazingdb
