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
  }

private:
  blazingdb::protocol::Client client;
};

}


}
}