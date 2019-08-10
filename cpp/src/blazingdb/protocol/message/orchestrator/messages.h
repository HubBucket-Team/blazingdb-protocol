#ifndef BLAZINGDB_PROTOCOL_MESSAGE_ORCHESTRATOR_MESSAGES_H
#define BLAZINGDB_PROTOCOL_MESSAGE_ORCHESTRATOR_MESSAGES_H

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>

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
        .port = pointer->nodeConnection()->port(),
        .path = std::string{pointer->nodeConnection()->path()->c_str()},
        .type = pointer->nodeConnection()->type()};
    calciteTime_ = pointer->calciteTime();
  };

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final {
    flatbuffers::FlatBufferBuilder builder{0};
    auto                           nodeInfo_offset = CreateNodeConnectionDirect(
        builder, nodeInfo.port, nodeInfo.path.data(), nodeInfo.type);
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


// DDLCreateTableRequestMessage

class NodeTableSchema {
public:
    explicit NodeTableSchema(const BlazingTableSchema & blazingTableSchema);

    explicit NodeTableSchema(const NodeTable * nodeTable);

    const BlazingTableSchema & blazingTable() const noexcept;

    flatbuffers::Offset<NodeTable> CreateFlatBufferItem(
        flatbuffers::FlatBufferBuilder & flatBufferBuilder) const noexcept;

private:
    std::shared_ptr<class NodeTableSchemaP> nodeTableSchemaP_;
};

class NodeTablesSchema {
public:
    explicit NodeTablesSchema();

    explicit NodeTablesSchema(
        const flatbuffers::Vector<flatbuffers::Offset<NodeTable>> * nodeTables);

    bool IsEmpty() const noexcept;

    const NodeTableSchema operator[](const std::size_t index) const noexcept;

    void Push(const NodeTableSchema & nodeTableSchema) noexcept;

    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NodeTable>>>
    CreateFlatBufferVector(
        flatbuffers::FlatBufferBuilder & flatBufferBuilder) const noexcept;

private:
    std::unique_ptr<class NodeTablesSchemaP> nodeTablesSchemaP_;
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

    schemaType = pointer->schemaType();

    blazingdb::protocol::BlazingTableSchema::Deserialize(pointer->gdf(), &gdf);

    auto files_list = pointer->files();
    for (const auto &file : (*files_list)) {
      files.push_back(std::string{file->c_str()});
    }

    csvDelimiter = std::string{pointer->csvDelimiter()->c_str()};

    csvLineTerminator = std::string{pointer->csvLineTerminator()->c_str()};

    csvSkipRows = pointer->csvSkipRows();
    resultToken = pointer->resultToken();

    nodeTables =
        std::make_unique<NodeTablesSchema>(pointer->nodeTables());
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder;
    auto name_offset = builder.CreateString(name);
    auto vectorOfColumnNames = builder.CreateVectorOfStrings(columnNames);
    auto vectorOfColumnTypes = builder.CreateVectorOfStrings(columnTypes);
    auto dbname_offset = builder.CreateString(dbName);
    auto gdf_offset = blazingdb::protocol::BlazingTableSchema::Serialize(builder, gdf);
    auto files_offset = builder.CreateVectorOfStrings(files);
    auto csvDelimiterOffset = builder.CreateString(csvDelimiter);
    auto csvLineTerminatorOffset = builder.CreateString(csvLineTerminator);

    builder.Finish(orchestrator::CreateDDLCreateTableRequest(
        builder,
        name_offset,
        vectorOfColumnNames,
        vectorOfColumnTypes,
        dbname_offset,
        schemaType,
        gdf_offset,
        files_offset,
        csvDelimiterOffset,
        csvLineTerminatorOffset,
        csvSkipRows,
        resultToken,
        nodeTables->CreateFlatBufferVector(builder)));

    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
public:
  std::string name;
  std::vector<std::string> columnNames;
  std::vector<std::string> columnTypes;
  std::string dbName;
  blazingdb::protocol::FileSchemaType schemaType;
	blazingdb::protocol::BlazingTableSchema gdf;
	std::vector<std::string> files;
  std::string csvDelimiter;
  std::string csvLineTerminator;
  uint32_t csvSkipRows;
  uint64_t resultToken;
  std::unique_ptr<NodeTablesSchema> nodeTables;
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

class DMLDistributedResponseMessage : public IMessage {
private:
    using NodeConnectionDTO = blazingdb::protocol::interpreter::NodeConnectionDTO;

public:
    DMLDistributedResponseMessage()
    : IMessage()
    { }

    DMLDistributedResponseMessage(const uint8_t* buffer)
    : IMessage() {
        auto pointer = flatbuffers::GetRoot<blazingdb::protocol::orchestrator::DMLDistributedResponse>(buffer);
        auto array_responses = pointer->responses();
        for (const auto& response : (*array_responses)) {
            auto resultToken = response->resultToken();
            auto nodeInfo  = NodeConnectionDTO {
                .port = response->nodeConnection()->port(),
                .path = std::string{response->nodeConnection()->path()->c_str()},
                .type = response->nodeConnection()->type()
            };
            auto calciteTime = response->calciteTime();
            responses.emplace_back(DMLResponseMessage(resultToken, nodeInfo, calciteTime));
        }
    }

    std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
        std::vector<flatbuffers::Offset<DMLResponse>> dml_responses;
        flatbuffers::FlatBufferBuilder builder;

        for (const auto& response : responses) {
            auto nodeInfo = CreateNodeConnectionDirect(builder,
                                                       response.getNodeInfo().port,
                                                       response.getNodeInfo().path.data(),
                                                       response.getNodeInfo().type);

            dml_responses.emplace_back(CreateDMLResponse(builder,
                                                         response.getResultToken(),
                                                         nodeInfo,
                                                         response.getCalciteTime()));
        }

        builder.Finish(CreateDMLDistributedResponseDirect(builder, &dml_responses));

        return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
    }

public:
    std::vector<DMLResponseMessage> responses;
};

} // orchestrator
} // protocol
} // blazingdb

#endif  // BLAZINGDB_PROTOCOL_MESSAGE_ORCHESTRATOR_MESSAGES_H
