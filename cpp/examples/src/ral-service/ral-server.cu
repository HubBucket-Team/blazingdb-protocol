
#include <iostream>
#include <map>
#include <blazingdb/protocol/api.h>

#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/messages.cuh>
#include "../gdf/GDFColumn.cuh"

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


  libgdf::gdf_column_cpp one;
  libgdf::create_sample_gdf_column(one); 
  libgdf::print_column(one.get_gdf_column());

  interpreter::BlazingMetadataDTO  metadata = {
    .status = "OK",
    .message = "metadata message",
    .time = 0.1f,
    .rows = 1
  }; 
  std::vector<std::string> fieldNames = {"id", "age"};
  std::vector<::libgdf::gdf_column> values = {
    ::libgdf::gdf_column {
                      .data = one.data(),
                      .valid = one.valid(),
                      .size = one.size(),
                      .dtype = (libgdf::gdf_dtype)one.dtype(),
                      .null_count = one.null_count(),
                      .dtype_info = libgdf::gdf_dtype_extra_info {
                          .time_unit = (libgdf::gdf_time_unit)0,
                      },
                  }, 
  };
  interpreter::GetResultResponseMessage responsePayload(metadata, fieldNames, values);
  return std::make_pair(Status_Success, responsePayload.getBufferData());
}


static result_pair executePlanService(uint64_t accessToken, Buffer&& requestPayloadBuffer)   {
  interpreter::ExecutePlanRequestMessage requestPayload(requestPayloadBuffer.data());

  // ExecutePlan
  std::cout << "accessToken: " << accessToken << std::endl;
  std::cout << "query: " << requestPayload.getLogicalPlan() << std::endl;
  std::cout << "tableGroup: " << requestPayload.getTableGroup().name << std::endl;
	std::cout << "tableSize: " << requestPayload.getTableGroup().tables.size() << std::endl;
	std::cout << "FirstColumnSize: "
			<< requestPayload.getTableGroup().tables[0].columns[0].size
			<< std::endl;

	libgdf::print_column(&requestPayload.getTableGroup().tables[0].columns[0]);

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
