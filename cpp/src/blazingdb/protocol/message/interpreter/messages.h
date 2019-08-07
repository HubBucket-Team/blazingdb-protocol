#ifndef BLAZINGDB_PROTOCOL_MESSAGE_INTERPRETER_MESSAGES_H
#define BLAZINGDB_PROTOCOL_MESSAGE_INTERPRETER_MESSAGES_H

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/utils.h>
#include <blazingdb/protocol/message/interpreter/gdf_dto.h>

namespace blazingdb {
namespace protocol {

namespace interpreter {


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
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(interpreter::CreateGetResultRequest(builder, value_));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  uint64_t  getResultToken () {
    return value_;
  }
};


struct NodeConnectionDTO {
  int port;
  std::string path;
  NodeConnectionType type;
};

class  ExecutePlanResponseMessage : public IMessage {
public:
  ExecutePlanResponseMessage (uint64_t resultToken, const NodeConnectionDTO &nodeInfo);

  ExecutePlanResponseMessage (const uint8_t* buffer);
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override ;

  uint64_t getResultToken();

  NodeConnectionDTO getNodeInfo() ;
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
  GetResultResponseMessage (const BlazingMetadataDTO&  metadata, const std::vector<std::string>& columnNames, const std::vector<uint64_t>& columnTokens, const std::vector<::gdf_dto::gdf_column>& columns);

  GetResultResponseMessage (const uint8_t* buffer);

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override ;

  BlazingMetadataDTO getMetadata();

  std::vector<std::string> getColumnNames();

  std::vector<uint64_t> getColumnTokens();

  std::vector<::gdf_dto::gdf_column> getColumns();
public:
  BlazingMetadataDTO  metadata;
  std::vector<std::string> columnNames;
  std::vector<uint64_t> columnTokens;
  std::vector<::gdf_dto::gdf_column> columns;
};


class  CreateTableResponseMessage : public IMessage {
public:
  CreateTableResponseMessage (const blazingdb::protocol::TableSchemaSTL& tableSchema);

  CreateTableResponseMessage (const uint8_t* buffer);
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override;
  const blazingdb::protocol::TableSchemaSTL& getTableSchema();

private:
  blazingdb::protocol::TableSchemaSTL tableSchema;
};


class RegisterDaskSliceRequestMessage : public IMessage {
public:
    using BlazingTableSchema = blazingdb::protocol::BlazingTableSchema;

    RegisterDaskSliceRequestMessage(
        const BlazingTableSchema & blazingTableSchema,
        const std::uint64_t        resultToken) noexcept;

    RegisterDaskSliceRequestMessage(const std::uint8_t * data) noexcept;

    std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final;

    const BlazingTableSchema & blazingTableSchema() const noexcept;
    std::uint64_t              resultToken() const noexcept;

private:
    const std::unique_ptr<const class RegisterDaskSliceRequestMessageP>
        registerDaskSliceRequestMessageP_;
};


class RegisterDaskSliceResponseMessage : public IMessage {
public:
    using BlazingTableSchema = blazingdb::protocol::BlazingTableSchema;

    RegisterDaskSliceResponseMessage(
        const BlazingTableSchema & blazingTableSchema) noexcept;

    RegisterDaskSliceResponseMessage(const std::uint8_t * data) noexcept;

    std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final;

    const BlazingTableSchema & blazingTableSchema() const noexcept;

private:
    const std::unique_ptr<const class RegisterDaskSliceResponseMessageP>
        registerDaskSliceResponseMessageP_;
};


} // interpreter
} // protocol
} // blazingdb

#endif  // BLAZINGDB_PROTOCOL_MESSAGE_INTERPRETER_MESSAGES_H
