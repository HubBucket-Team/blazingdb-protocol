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


class DMLResponseMessage : public StringTypeMessage<orchestrator::DMLResponse> {
public:  
  
  DMLResponseMessage(const std::string& string_value) 
    : StringTypeMessage<orchestrator::DMLResponse>(string_value) 
  {
  }
  
  DMLResponseMessage (const uint8_t* buffer) 
    :  StringTypeMessage<orchestrator::DMLResponse>(buffer, &orchestrator::DMLResponse::resultToken)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDMLResponseDirect);
  }

  std::string getToken () {
    return string_value;
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


class DDLResponseMessage : public StringTypeMessage<orchestrator::DMLResponse> {
public:

  DDLResponseMessage(const std::string& string_value = "")
      : StringTypeMessage<orchestrator::DMLResponse>(string_value)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDMLResponseDirect);
  }
};

} // orchestrator
} // protocol
} // blazingdb
