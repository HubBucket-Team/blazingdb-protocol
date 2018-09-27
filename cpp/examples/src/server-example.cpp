#include <iostream>

#include <blazingdb/protocol/api.h>

using namespace blazingdb::protocol;

int main() {
  blazingdb::protocol::UnixSocketConnection connection({"/tmp/socket", std::allocator<char>()});
  blazingdb::protocol::Server server(connection);

  // only DML
  server.handle([](const blazingdb::protocol::Buffer &requestBuffer)
                    -> blazingdb::protocol::Buffer {

    RequestMessage request{requestBuffer.data()};
    DMLRequestMessage requestPayload(request.getPayloadBuffer());

    std::cout << "header: " << request.header() << std::endl;
    std::cout << "query: " << requestPayload.getQuery() << std::endl;

    std::string logicalPlan = "LogicalUnion(all=[false])";

    DMLResponseMessage responsePayload{logicalPlan};
    ResponseMessage responseObject{Status_Success, responsePayload};
    auto bufferedData = responseObject.getBufferData();
    Buffer buffer{bufferedData->data(),
                bufferedData->size()};
    return buffer;
  });
  return 0;
}
