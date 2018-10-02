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




class DMLResponseMessage : public StringTypeMessage<interpreter::DMLResponse> {
public:

  DMLResponseMessage(const std::string& string_value)
      : StringTypeMessage<interpreter::DMLResponse>(string_value)
  {
  }

  DMLResponseMessage (const uint8_t* buffer)
      :  StringTypeMessage<interpreter::DMLResponse>(buffer, &interpreter::DMLResponse::resultToken)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(interpreter::CreateDMLResponseDirect);
  }

  std::string getToken () {
    return string_value;
  }
};


} // interpreter
} // protocol
} // blazingdb
