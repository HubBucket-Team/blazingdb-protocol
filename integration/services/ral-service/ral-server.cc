#include <iostream>

#include <blazingdb/protocol/api.h>

#include <blazingdb/protocol/interpreter/messages.h>

<<<<<<< HEAD:integration/services/ral-service/src/ral-server.cc
namespace blazingdb
{
namespace protocol
{
namespace interpreter
{

auto InterpreterService(const blazingdb::protocol::Buffer &requestBuffer)
    -> blazingdb::protocol::Buffer
{
  RequestMessage request{requestBuffer.data()};
  std::cout << "header: " << static_cast<int>(request.messageType())
            << std::endl;

  if (request.messageType() == interpreter::MessageType_ExecutePlan)
    {
      DMLRequestMessage requestPayload(request.getPayloadBuffer());

      std::cout << "query: " << requestPayload.getLogicalPlan() << std::endl;

      uint64_t token = 543210L;

      DMLResponseMessage responsePayload{token};
      ResponseMessage responseObject{Status_Success, responsePayload};
      auto bufferedData = responseObject.getBufferData();
      Buffer buffer{bufferedData->data(), bufferedData->size()};
      return buffer;
    }
  else if (request.messageType() == interpreter::MessageType_GetResult)
    {
      flatbuffers::FlatBufferBuilder builder;
      std::vector<std::string> names{"iron", "man"};
      auto vectorOfNames = builder.CreateVectorOfStrings(names);
      builder.Finish(CreateGetResultResponse(builder, vectorOfNames));
      std::shared_ptr<flatbuffers::DetachedBuffer> payload =
          std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
      ResponseMessage responseMessage(Status::Status_Success, payload);
      std::shared_ptr<flatbuffers::DetachedBuffer> response =
          responseMessage.getBufferData();
      return Buffer{response->data(), response->size()};
    }
}

}  // namespace interpreter
}  // namespace protocol
}  // namespace blazingdb

using namespace blazingdb::protocol::interpreter;

int main()
{
  blazingdb::protocol::UnixSocketConnection connection(
      {"/tmp/ral.socket", std::allocator<char>()});
  blazingdb::protocol::Server server(connection);


  server.handle(InterpreterService);

  return 0;
}
