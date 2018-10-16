#pragma once 

#include <iostream>
#include <string>
#include <exception>

#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"

#include <blazingdb/protocol/message/interpreter/messages.cuh>

namespace blazingdb {
namespace protocol { 
namespace interpreter {
 
class InterpreterClient {
public:
  InterpreterClient(blazingdb::protocol::Connection & connection) : client {connection}
  {}
 
 
  uint64_t executePlan(std::string logicalPlan, const blazingdb::protocol::TableGroup *tableGroup, int64_t access_token)  {
    
    return 1L;
  }

  uint64_t executePlan(std::string logicalPlan, const ::blazingdb::protocol::TableGroupDTO &tableGroup, int64_t access_token)  {
    auto bufferedData = MakeRequest(interpreter::MessageType_ExecutePlan,
                                     access_token,
                                     ExecutePlanRequestMessage{logicalPlan, tableGroup});

    Buffer responseBuffer = client.send(bufferedData);
    ResponseMessage response{responseBuffer.data()};
    
    if (response.getStatus() == Status_Error) {
      ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
      throw std::runtime_error(errorMessage.getMessage());
    }
    ExecutePlanResponseMessage responsePayload(response.getPayloadBuffer());
    return responsePayload.getResultToken();
  }

  std::vector<::libgdf::gdf_column> getResult(uint64_t resultToken, int64_t access_token){
    auto bufferedData = MakeRequest(interpreter::MessageType_GetResult,
                                     access_token,
                                     interpreter::GetResultRequestMessage {resultToken});

    Buffer responseBuffer = client.send(bufferedData);
    ResponseMessage response{responseBuffer.data()};
    
    if (response.getStatus() == Status_Error) {
      ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
      throw std::runtime_error(errorMessage.getMessage());
    }
    std::cout << "get_result_status: " << response.getStatus() << std::endl;

    interpreter::GetResultResponseMessage responsePayload(response.getPayloadBuffer());
    std::cout << "getValues: " << responsePayload.getMetadata().message << std::endl;

    return responsePayload.getValues();
  }

  Status closeConnection (int64_t access_token) {
    auto bufferedData = MakeRequest(interpreter::MessageType_CloseConnection,
                                    access_token,
                                    ZeroMessage{});
    Buffer responseBuffer = client.send(bufferedData);
    ResponseMessage response{responseBuffer.data()};
    if (response.getStatus() == Status_Error) {
      ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
      throw std::runtime_error(errorMessage.getMessage());
    }
    return response.getStatus();
  }

protected:
  blazingdb::protocol::Client client;
};


} // interpreter
} // protocol
} // blazingdb
