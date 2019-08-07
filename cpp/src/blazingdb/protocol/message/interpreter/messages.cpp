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


  // RegisterDaskSliceRequestMessage

  class RegisterDaskSliceRequestMessageP {
  public:
      virtual std::shared_ptr<flatbuffers::DetachedBuffer>
      getBufferData() const = 0;

      virtual const BlazingTableSchema & blazingTableSchema() const
          noexcept = 0;

      virtual std::uint64_t resultToken() const noexcept = 0;
  };

  class RegisterDaskSliceRequestMessagePDeserialized
      : public RegisterDaskSliceRequestMessageP {
  public:
      explicit RegisterDaskSliceRequestMessagePDeserialized(
          const BlazingTableSchema & blazingTableSchema,
          const std::uint64_t        resultToken)
          : blazingTableSchema_{blazingTableSchema}, resultToken_{resultToken} {
      }

      std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final {
          flatbuffers::FlatBufferBuilder flatBufferBuilder;

          flatbuffers::Offset<BlazingTable> blazingTable =
              BlazingTableSchema::Serialize(flatBufferBuilder,
                                            blazingTableSchema_);

          flatBufferBuilder.Finish(CreateRegisterDaskSliceRequest(
              flatBufferBuilder, blazingTable, resultToken_));

          return std::make_shared<flatbuffers::DetachedBuffer>(
              flatBufferBuilder.Release());
      }

      const BlazingTableSchema & blazingTableSchema() const noexcept final {
          return blazingTableSchema_;
      }

      std::uint64_t resultToken() const noexcept final { return resultToken_; }

  private:
      const BlazingTableSchema & blazingTableSchema_;
      const std::uint64_t        resultToken_;
  };

  class RegisterDaskSliceRequestMessagePSerialized
      : public RegisterDaskSliceRequestMessageP {
  public:
      explicit RegisterDaskSliceRequestMessagePSerialized(
          const std::uint8_t * data)
          : data_{data} {
          using blazingdb::protocol::interpreter::RegisterDaskSliceRequest;
          const RegisterDaskSliceRequest * registerDaskSliceRequestRoot =
              flatbuffers::GetRoot<RegisterDaskSliceRequest>(data);

          using blazingdb::protocol::interpreter::RegisterDaskSliceRequestT;
          flatbuffers::unique_ptr<RegisterDaskSliceRequestT>
              registerDaskSliceRequestT =
                  flatbuffers::unique_ptr<RegisterDaskSliceRequestT>(
                      registerDaskSliceRequestRoot->UnPack());

          using blazingdb::protocol::BlazingTable;
          const BlazingTable * blazingTable =
              registerDaskSliceRequestRoot->table();

          BlazingTableSchema::Deserialize(blazingTable, &blazingTableSchema_);
          resultToken_ = registerDaskSliceRequestT->resultToken;
      }

      std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final {
          return std::make_shared<flatbuffers::DetachedBuffer>(
              nullptr, false, const_cast<std::uint8_t *>(data_), 0, nullptr, 0);
      }

      const BlazingTableSchema & blazingTableSchema() const noexcept final {
          return blazingTableSchema_;
      }

      std::uint64_t resultToken() const noexcept final { return resultToken_; }

  private:
      BlazingTableSchema   blazingTableSchema_;
      std::uint64_t        resultToken_;
      const std::uint8_t * data_;
  };

  RegisterDaskSliceRequestMessage::RegisterDaskSliceRequestMessage(
      const BlazingTableSchema & blazingTableSchema,
      const std::uint64_t        resultToken) noexcept
      : registerDaskSliceRequestMessageP_{
            std::make_unique<RegisterDaskSliceRequestMessagePDeserialized>(
                blazingTableSchema, resultToken)} {}

  RegisterDaskSliceRequestMessage::RegisterDaskSliceRequestMessage(
      const std::uint8_t * data) noexcept
      : registerDaskSliceRequestMessageP_{
            std::make_unique<RegisterDaskSliceRequestMessagePSerialized>(
                data)} {}

  std::shared_ptr<flatbuffers::DetachedBuffer>
  RegisterDaskSliceRequestMessage::getBufferData() const {
      return registerDaskSliceRequestMessageP_->getBufferData();
  }

  const BlazingTableSchema &
  RegisterDaskSliceRequestMessage::blazingTableSchema() const noexcept {
      return registerDaskSliceRequestMessageP_->blazingTableSchema();
  }

  std::uint64_t RegisterDaskSliceRequestMessage::resultToken() const noexcept {
      return registerDaskSliceRequestMessageP_->resultToken();
  }


  // RegisterDaskSliceResponseMessage

  class RegisterDaskSliceResponseMessageP {
  public:
      virtual std::shared_ptr<flatbuffers::DetachedBuffer>
      getBufferData() const = 0;

      virtual const BlazingTableSchema & blazingTableSchema() const
          noexcept = 0;
  };

  class RegisterDaskSliceResponseMessagePDeserialized
      : public RegisterDaskSliceResponseMessageP {
  public:
      explicit RegisterDaskSliceResponseMessagePDeserialized(
          const BlazingTableSchema & blazingTableSchema)
          : blazingTableSchema_{blazingTableSchema} {}

      std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final {
          flatbuffers::FlatBufferBuilder flatBufferBuilder;

          flatbuffers::Offset<BlazingTable> blazingTable =
              BlazingTableSchema::Serialize(flatBufferBuilder,
                                            blazingTableSchema_);

          flatBufferBuilder.Finish(
              CreateRegisterDaskSliceResponse(flatBufferBuilder, blazingTable));

          return std::make_shared<flatbuffers::DetachedBuffer>(
              flatBufferBuilder.Release());
      }

      const BlazingTableSchema & blazingTableSchema() const noexcept final {
          return blazingTableSchema_;
      }

  private:
      const BlazingTableSchema & blazingTableSchema_;
  };

  class RegisterDaskSliceResponseMessagePSerialized
      : public RegisterDaskSliceResponseMessageP {
  public:
      explicit RegisterDaskSliceResponseMessagePSerialized(
          const std::uint8_t * data)
          : data_{data} {
          using blazingdb::protocol::interpreter::RegisterDaskSliceResponse;
          const RegisterDaskSliceResponse * registerDaskSliceResponseRoot =
              flatbuffers::GetRoot<RegisterDaskSliceResponse>(data);

          using blazingdb::protocol::interpreter::RegisterDaskSliceResponseT;
          flatbuffers::unique_ptr<RegisterDaskSliceResponseT>
              registerDaskSliceResponseT =
                  flatbuffers::unique_ptr<RegisterDaskSliceResponseT>(
                      registerDaskSliceResponseRoot->UnPack());

          using blazingdb::protocol::BlazingTable;
          const BlazingTable * blazingTable =
              registerDaskSliceResponseRoot->table();

          BlazingTableSchema::Deserialize(blazingTable, &blazingTableSchema_);
      }

      std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const final {
          return std::make_shared<flatbuffers::DetachedBuffer>(
              nullptr, false, const_cast<std::uint8_t *>(data_), 0, nullptr, 0);
          ;
      }

      const BlazingTableSchema & blazingTableSchema() const noexcept final {
          return blazingTableSchema_;
      }

  private:
      BlazingTableSchema   blazingTableSchema_;
      const std::uint8_t * data_;
  };

  RegisterDaskSliceResponseMessage::RegisterDaskSliceResponseMessage(
      const BlazingTableSchema & blazingTableSchema) noexcept
      : registerDaskSliceResponseMessageP_{
            std::make_unique<RegisterDaskSliceResponseMessagePDeserialized>(
                blazingTableSchema)} {}

  RegisterDaskSliceResponseMessage::RegisterDaskSliceResponseMessage(
      const std::uint8_t * data) noexcept
      : registerDaskSliceResponseMessageP_{
            std::make_unique<RegisterDaskSliceResponseMessagePSerialized>(
                data)} {}

  std::shared_ptr<flatbuffers::DetachedBuffer>
  RegisterDaskSliceResponseMessage::getBufferData() const {
      return registerDaskSliceResponseMessageP_->getBufferData();
  }

  const BlazingTableSchema &
  RegisterDaskSliceResponseMessage::blazingTableSchema() const noexcept {
      return registerDaskSliceResponseMessageP_->blazingTableSchema();
  }

} // interpreter
} // protocol
} // blazingdb
