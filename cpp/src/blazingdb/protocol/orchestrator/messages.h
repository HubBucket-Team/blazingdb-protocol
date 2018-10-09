#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"

namespace blazingdb {
namespace protocol {
namespace orchestrator {


class DMLRequestMessage : public StringTypeMessage<orchestrator::DMLRequest> {
public:
  DMLRequestMessage(const std::string& string_value)
    : StringTypeMessage<orchestrator::DMLRequest>(string_value) 
  {
  }
  
  DMLRequestMessage (const uint8_t* buffer) 
    :  StringTypeMessage<orchestrator::DMLRequest>(buffer, &orchestrator::DMLRequest::query)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDMLRequestDirect);
  }

  std::string getQuery () {
    return string_value;
  }
};


class DMLResponseMessage : public TypedMessage<uint64_t, orchestrator::DMLResponse> {
public:  
  
  DMLResponseMessage(const std::uint64_t& value)
    : TypedMessage<uint64_t, orchestrator::DMLResponse>(value)
  {
  }
  
  DMLResponseMessage (const uint8_t* buffer) 
    :  TypedMessage<uint64_t, orchestrator::DMLResponse>(buffer, &orchestrator::DMLResponse::resultToken)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDMLResponse);
  }

  uint64_t  getToken () {
    return value_;
  }
};


class DDLRequestMessage : public StringTypeMessage<orchestrator::DMLRequest> {
public:
  DDLRequestMessage(const std::string& string_value)
      : StringTypeMessage<orchestrator::DMLRequest>(string_value)
  {
  }

  DDLRequestMessage (const uint8_t* buffer)
      :  StringTypeMessage<orchestrator::DMLRequest>(buffer, &orchestrator::DMLRequest::query)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDMLRequestDirect);
  }

  std::string getQuery () {
    return string_value;
  }
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

    // std::vector<std::string> columnNames{"name", "surname"};
    auto vectorOfColumnNames = builder.CreateVectorOfStrings(columnNames);
    // std::vector<std::string> columnTypes{"string", "string"};
    auto vectorOfColumnTypes = builder.CreateVectorOfStrings(columnTypes);

    auto dbname_offset = builder.CreateString(dbName);

    builder.Finish(orchestrator::CreateDDLCreateTableRequest(builder, name_offset,vectorOfColumnNames, vectorOfColumnTypes, dbname_offset));

    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
private:
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
