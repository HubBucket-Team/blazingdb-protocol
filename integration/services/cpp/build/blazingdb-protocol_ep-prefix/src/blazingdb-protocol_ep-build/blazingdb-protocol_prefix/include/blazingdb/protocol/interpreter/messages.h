#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"

namespace blazingdb {
namespace protocol {
namespace interpreter {


class DMLRequestMessage : public StringTypeMessage<interpreter::DMLRequest> {
public:
  DMLRequestMessage(const std::string& string_value)
      : StringTypeMessage<interpreter::DMLRequest>(string_value)
  {
  }

  DMLRequestMessage (const uint8_t* buffer)
      :  StringTypeMessage<interpreter::DMLRequest>(buffer, &interpreter::DMLRequest::logicalPlan)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    return this->getBufferDataUsing(interpreter::CreateDMLRequestDirect);
  }

  std::string getLogicalPlan() {
    return string_value;
  }
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

//class GetResultResponseMessage : public IMessage {
  //
//};

class DMLResponseMessage : public TypedMessage<uint64_t, interpreter::DMLResponse> {
public:

  DMLResponseMessage(const uint64_t& value)
      : TypedMessage<uint64_t, interpreter::DMLResponse>(value)
  {
  }

  DMLResponseMessage (const uint8_t* buffer)
      :  TypedMessage<uint64_t, interpreter::DMLResponse>(buffer, &interpreter::DMLResponse::resultToken)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(interpreter::CreateDMLResponse);
  }

  uint64_t getToken () {
    return value_;
  }
};


} // interpreter
} // protocol
} // blazingdb
