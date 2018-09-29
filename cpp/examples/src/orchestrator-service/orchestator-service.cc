#include <iostream>

#include <blazingdb/protocol/api.h>

#include "calcite-client.h"
#include "ral-client.h"

#include "orchestrator-server.h"

 

using namespace blazingdb::protocol;


int main() {
  blazingdb::protocol::UnixSocketConnection server_connection({"/tmp/orchestrator.socket", std::allocator<char>()});
  blazingdb::protocol::Server server(server_connection);


  auto orchestratorService = [](const blazingdb::protocol::Buffer &requestBuffer) -> blazingdb::protocol::Buffer {
    RequestMessage request{requestBuffer.data()};
    orchestrator::DMLRequestMessage requestPayload(request.getPayloadBuffer());

    std::cout << "header: " << (int)request.header() << std::endl;
    std::cout << "query: " << requestPayload.getQuery() << std::endl;
    std::string token = "null_token";

    if ( request.header() == orchestrator::MessageType_DML) {
      auto query = requestPayload.getQuery();
      try {
        blazingdb::protocol::UnixSocketConnection calcite_client_connection{"/tmp/calcite.socket"};
        calcite::CalciteClient calcite_client{calcite_client_connection};
        auto logicalPlan = calcite_client.getLogicalPlan(query);
        std::cout << "plan:" << logicalPlan << std::endl;

        try {
          blazingdb::protocol::UnixSocketConnectigon ral_client_connection{"/tmp/ral.socket"};
          interpreter::InterpreterClient ral_client{ral_client_connection};
          token = ral_client.executePlan(logicalPlan);
          std::cout << "token:" << token << std::endl;
        } catch (std::runtime_error &error) {
          std::cout << error.what() << std::endl;
        }
      } catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
      }
    }
    orchestrator::DMLResponseMessage responsePayload{token};
    std::cout << responsePayload.getToken() << std::endl;
    ResponseMessage responseObject{Status_Success, responsePayload};
    auto bufferedData = responseObject.getBufferData();
    Buffer buffer{bufferedData->data(),
                  bufferedData->size()};
    return buffer;
  };

  server.handle(orchestratorService);

  return 0;
} 

