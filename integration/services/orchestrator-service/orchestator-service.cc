#include <iostream>
#include <map>
#include <tuple>
#include <blazingdb/protocol/api.h>

#include "calcite-client.h"
#include "ral-client.h"

#include "orchestrator-server.h"

#include <blazingdb/protocol/authorization/messages.h>


using namespace blazingdb::protocol;
using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;

static result_pair  authorizationService(const uint8_t* buffer)  {
  int64_t token = 123456789L;
  authorization::AuthResponseMessage response{token};
  std::cout << "authorizationService: " << token << std::endl;
  return std::make_pair(Status_Success, response.getBufferData());
};

static result_pair  dmlService(const uint8_t* buffer)  {
  orchestrator::DMLRequestMessage requestPayload(buffer);
  auto query = requestPayload.getQuery();
  std::cout << "DML: " << query << std::endl;
  std::string token = "null_token";

  try {
    blazingdb::protocol::UnixSocketConnection calcite_client_connection{"/tmp/calcite.socket"};
    calcite::CalciteClient calcite_client{calcite_client_connection};
    auto logicalPlan = calcite_client.getLogicalPlan(query);
    std::cout << "plan:" << logicalPlan << std::endl;
    try {
      blazingdb::protocol::UnixSocketConnection ral_client_connection{"/tmp/ral.socket"};
      interpreter::InterpreterClient ral_client{ral_client_connection};
      token = ral_client.executePlan(logicalPlan);
      std::cout << "token:" << token << std::endl;
    } catch (std::runtime_error &error) {
      // error with query plan: not token
      std::cout << error.what() << std::endl;
      ResponseErrorMessage errorMessage{ std::string{error.what()} };
      return std::make_pair(Status_Error, errorMessage.getBufferData());
    }
  } catch (std::runtime_error &error) {
    // error with query: not logical plan error
    std::cout << error.what() << std::endl;
    ResponseErrorMessage errorMessage{ std::string{error.what()} };
    return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  orchestrator::DMLResponseMessage response{token};
  return std::make_pair(Status_Success, response.getBufferData());
};

static result_pair ddlService(const uint8_t* buffer)  {
  orchestrator::DDLRequestMessage requestPayload(buffer);
  auto query = requestPayload.getQuery();
  std::cout << "DDL: " << query << std::endl;
   try {
    blazingdb::protocol::UnixSocketConnection calcite_client_connection{"/tmp/calcite.socket"};
    calcite::CalciteClient calcite_client{calcite_client_connection};
    auto status = calcite_client.updateSchema(query);
    std::cout << "status:" << status << std::endl;
  } catch (std::runtime_error &error) {
     // error with ddl query
     std::cout << error.what() << std::endl;
     ResponseErrorMessage errorMessage{ std::string{error.what()} };
     return std::make_pair(Status_Error, errorMessage.getBufferData());
  }
  orchestrator::DDLResponseMessage response{""};
  return std::make_pair(Status_Success, response.getBufferData());
};

using FunctionType = result_pair (*)(const uint8_t* buffer);

int main() {
  blazingdb::protocol::UnixSocketConnection server_connection({"/tmp/orchestrator.socket", std::allocator<char>()});
  blazingdb::protocol::Server server(server_connection);


  std::map<int8_t, FunctionType> services;
  services.insert(std::make_pair(orchestrator::MessageType_DML, &dmlService));
  services.insert(std::make_pair(orchestrator::MessageType_DDL, &ddlService));
  services.insert(std::make_pair(authorization::MessageType_AuthOpen, &authorizationService));

  auto orchestratorService = [&services](const blazingdb::protocol::Buffer &requestBuffer) -> blazingdb::protocol::Buffer {
    RequestMessage request{requestBuffer.data()};
    std::cout << "header: " << (int)request.messageType() << std::endl;

    auto result = services[request.messageType()] ( request.getPayloadBuffer() );
    ResponseMessage responseObject{result.first, result.second};
    auto bufferedData = responseObject.getBufferData();
    Buffer buffer{bufferedData->data(),
                  bufferedData->size()};
    return buffer;
  };
  server.handle(orchestratorService);
  return 0;
}