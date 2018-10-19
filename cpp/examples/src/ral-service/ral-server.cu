
#include <iostream>
#include <map>
#include <blazingdb/protocol/api.h>

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.h>

#include "gdf/container/gdf_vector.cuh"
#include "gdf/util/gdf_utils.cuh"

using namespace blazingdb::protocol;

using result_pair = std::pair<Status, std::shared_ptr<flatbuffers::DetachedBuffer>>;
using FunctionType = result_pair (*)(uint64_t, Buffer&& buffer);



static result_pair closeConnectionService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
  std::cout << "accessToken: " << accessToken << std::endl;
  // remove from repository using accessToken

  ZeroMessage response{};
  return std::make_pair(Status_Success, response.getBufferData());
}

static result_pair getResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
   std::cout << "accessToken: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage request(requestPayloadBuffer.data());
  std::cout << "resultToken: " << request.getResultToken() << std::endl;


  ::gdf::container::GdfVector one;
  ::gdf::util::create_sample_gdf_column(one); 
  ::gdf::util::print_gdf_column(one.get_gdf_column());

  interpreter::BlazingMetadataDTO  metadata = {
    .status = "OK",
    .message = "metadata message",
    .time = 0.1f,
    .rows = 1
  }; 
  std::vector<std::string> fieldNames = {"id_column"};
  std::vector<::gdf_dto::gdf_column> values = {
    ::gdf_dto::gdf_column {
                      .data = ::gdf::util::BuildCudaIpcMemHandler(one.data()),
                      .valid = ::gdf::util::BuildCudaIpcMemHandler(one.valid()),
                      .size = one.size(),
                      .dtype = (gdf_dto::gdf_dtype)one.dtype(),
                      .null_count = one.null_count(),
                      .dtype_info = gdf_dto::gdf_dtype_extra_info {
                          .time_unit = (gdf_dto::gdf_time_unit)0,
                      },
                  }, 
  };
  interpreter::GetResultResponseMessage responsePayload(metadata, fieldNames, values);
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}


static result_pair freeResultService(uint64_t accessToken, Buffer&& requestPayloadBuffer) {
   std::cout << "freeResultService: " << accessToken << std::endl;

  interpreter::GetResultRequestMessage request(requestPayloadBuffer.data());
  std::cout << "resultToken: " << request.getResultToken() << std::endl;
  
  ZeroMessage response{};
  return std::make_pair(Status_Success, response.getBufferData());
}


static result_pair executePlanService(uint64_t accessToken, Buffer&& requestPayloadBuffer)   {
  interpreter::ExecutePlanRequestMessage requestPayload(requestPayloadBuffer.data());

  // ExecutePlan
  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.getLogicalPlan() << std::endl;
  std::cout << "tableGroup: " << requestPayload.getTableGroup().name << std::endl;
	std::cout << "tableSize: " << requestPayload.getTableGroup().tables.size() << std::endl;

  ::gdf::util::ToBlazingFrame(requestPayload.getTableGroup());

  uint64_t resultToken = 543210L;
  interpreter::NodeConnectionInformationDTO nodeInfo {
      .path = "/tmp/ral.socket",
      .type = interpreter::NodeConnectionType {interpreter::NodeConnectionType_IPC}
  };
  interpreter::ExecutePlanResponseMessage responsePayload{resultToken, nodeInfo};
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}


int main() {
  blazingdb::protocol::UnixSocketConnection connection({"/tmp/ral.socket", std::allocator<char>()});
  blazingdb::protocol::Server server(connection);

  std::map<int8_t, FunctionType> services;
  services.insert(std::make_pair(interpreter::MessageType_ExecutePlan, &executePlanService));
  services.insert(std::make_pair(interpreter::MessageType_CloseConnection, &closeConnectionService));
  services.insert(std::make_pair(interpreter::MessageType_GetResult, &getResultService));
  services.insert(std::make_pair(interpreter::MessageType_FreeResult, &freeResultService));

  auto interpreterServices = [&services](const blazingdb::protocol::Buffer &requestPayloadBuffer) -> blazingdb::protocol::Buffer {
    RequestMessage request{requestPayloadBuffer.data()};
    std::cout << "header: " << (int)request.messageType() << std::endl;

    auto result = services[request.messageType()] ( request.accessToken(),  request.getPayloadBuffer() );
    ResponseMessage responseObject{result.first, result.second};
    return Buffer{responseObject.getBufferData()};
  };
  server.handle(interpreterServices);

  return 0;
}
