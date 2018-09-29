#pragma once 

#include <iostream>
#include <exception>

#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"

#include <blazingdb/protocol/calcite/messages.h>

namespace blazingdb {
namespace protocol { 

namespace calcite { 

class CalciteClient {
public:
  CalciteClient(blazingdb::protocol::Connection& con) : client { con }
  {}

  //todo: reducir codigo usando MakeRequest & MakeResponse
  std::string getLogicalPlan(std::string query)  {
    DMLRequestMessage requestPayload{query};
    RequestMessage requestObject{calcite::MessageType_DML, requestPayload}; 
    
    auto bufferedData = requestObject.getBufferData();

    Buffer buffer{bufferedData->data(), 
                  bufferedData->size()};

    Buffer responseBuffer = client.send(buffer);
    ResponseMessage response{responseBuffer.data()};
    if (response.getStatus() == Status_Error) {
      ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
      throw std::runtime_error(errorMessage.getMessage());
    }
    DMLResponseMessage responsePayload(response.getPayloadBuffer());
    return responsePayload.getLogicalPlan();
  }

  //todo: reducir codigo usando MakeRequest & MakeResponse
  Status updateSchema(std::string statement)    {
    DDLRequestMessage requestPayload{statement};
    RequestMessage requestObject{calcite::MessageType_DDL, requestPayload}; 
    
    auto bufferedData = requestObject.getBufferData();

    Buffer buffer{bufferedData->data(), 
                  bufferedData->size()};

    Buffer responseBuffer = client.send(buffer);
    ResponseMessage response{responseBuffer.data()};
    if (response.getStatus() == Status_Error) {
      ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
      throw std::runtime_error(errorMessage.getMessage());
    }
    return response.getStatus();
  }

private:
  blazingdb::protocol::Client client;
};


}
}
}
