#pragma once 

#include <iostream>
#include <string>
#include <exception>

#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"

#include <blazingdb/protocol/interpreter/messages.h>

namespace blazingdb {
namespace protocol { 



namespace interpreter {
 
class InterpreterClient {
public:
  InterpreterClient(blazingdb::protocol::Connection & connection) : client {connection}
  {}

  std::string executePlan(std::string logicalPlan)  {
    int64_t sessionToken = 0;
    auto bufferedData = MakeRequest(interpreter::MessageType_ExecutePlan,
                                     logicalPlan.length(),
                                     sessionToken,
                                     DMLRequestMessage{logicalPlan});
    Buffer responseBuffer = client.send(bufferedData);
    auto response = MakeResponse<DMLResponseMessage>(responseBuffer);
    return response.getToken();

    // DMLRequestMessage requestPayload{logicalPlan};

    // RequestMessage requestObject{interpreter::MessageType_ExecutePlan, requestPayload};

    // auto bufferedData = requestObject.getBufferData();

    // std::cout << "\t1.1\n";
    // Buffer responseBuffer = client.send(bufferedData);
    // std::cout << "\t1.2\n";

    // ResponseMessage response{responseBuffer.data()};
    // if (response.getStatus() == Status_Error) {
    //   ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
    //   throw std::runtime_error(errorMessage.getMessage());
    // }
    // DMLResponseMessage responsePayload(response.getPayloadBuffer());
    // return responsePayload.getToken();

  }

private:
  blazingdb::protocol::Client client;
};

}


}
}