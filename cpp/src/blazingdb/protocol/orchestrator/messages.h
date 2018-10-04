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
