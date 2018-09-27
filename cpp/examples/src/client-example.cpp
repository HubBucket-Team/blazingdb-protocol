#include <iostream>

#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "protocol_generated.h"


namespace blazingdb {
namespace protocol {


class IMessage {
public:
  IMessage() = default;

  virtual ~IMessage() = default;

  virtual std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const = 0;

};


class ResponseMessage  : public IMessage {
public:  
  ResponseMessage (const uint8_t* buffer) 
      : IMessage()
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Response>(buffer);
      status = pointer->status();
      payloadBuffer = (uint8_t*)pointer->payload()->data();
  }
  // ResponseMessage(Status status, IMessage& payload) : IMessage() {
  //     status = status;
  //     _copy_payload = payload.getBufferData();
  // }

  // std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
  //   flatbuffers::FlatBufferBuilder builder{1024};

  //   auto payload_offset = builder.CreateVector(payloadBuffer.data(), payloadBuffer.size());
  //   auto root_offset = CreateResponse(builder, status, payload_offset);
  //   builder.Finish(root_offset);
  //   return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  // }
  
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {

      return nullptr;
  }

  const uint8_t* getPayloadBuffer() {
    return payloadBuffer;
  }
private:
    Status          status;
    uint8_t*  payloadBuffer;

 };


class DMLResponseMessage : public IMessage {
public:  

  DMLResponseMessage(const std::string& logicalPlan) : IMessage(), logicalPlan (logicalPlan){
  }
  
  DMLResponseMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::calcite::DMLResponse>(buffer);
    logicalPlan = std::string{pointer->logicalPlan()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto string_offset = builder.CreateString(logicalPlan);
    auto root_offset = calcite::CreateDMLResponse(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getLogicalPlan () {
    return logicalPlan;
  }
  
private:
  std::string logicalPlan;
};



class RequestMessage : public IMessage {
public:  
  RequestMessage (const uint8_t* buffer) : IMessage() {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer);
      headerType = pointer->header();
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
      
  }
  RequestMessage(int8_t headerType, IMessage& payload) : IMessage() {
      headerType = headerType;
      _copy_payload = payload.getBufferData(); 
   
      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{1024};

    auto payload_offset = builder.CreateVector(payloadBuffer, payloadBufferSize);
    auto root_offset = CreateRequest(builder, headerType, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  const uint8_t* getPayloadBuffer() {
    return payloadBuffer;
  }

  size_t getPayloadBufferSize() {
    return  payloadBufferSize;
  }

private:
    int8_t             headerType;
    const uint8_t*    payloadBuffer;
    size_t            payloadBufferSize;

    std::shared_ptr<flatbuffers::DetachedBuffer>  _copy_payload; 
};


class DMLRequestMessage : public IMessage {
public: 

  DMLRequestMessage(const std::string& query) : IMessage(), query (query){
  }
  
  DMLRequestMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::calcite::DMLRequest>(buffer);
    query = std::string{pointer->query()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto string_offset = builder.CreateString(query);
    auto root_offset = calcite::CreateDMLRequest(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

private:
  std::string query;
};

class CalciteClient {
public:
  CalciteClient(blazingdb::protocol::Connection & connection) : client {connection}
  {}

  std::string getLogicalPlan(std::string query) {
    DMLRequestMessage requestPayload{query};
    RequestMessage requestObject{calcite::MessageType_DML, requestPayload}; 
    
    auto bufferedData = requestObject.getBufferData();

    Buffer buffer{bufferedData->data(), 
                  bufferedData->size()};

    Buffer responseBuffer = client.send(buffer);
    ResponseMessage response{responseBuffer.data()};
    DMLResponseMessage responsePayload(response.getPayloadBuffer());
    return responsePayload.getLogicalPlan();
  }

  void updateSchema(std::string query) {
      DMLRequestMessage requestPayload{query};
      RequestMessage requestObject{calcite::MessageType_DDL, requestPayload}; 
      
      auto bufferedData = requestObject.getBufferData();

      Buffer buffer{bufferedData->data(), 
                    bufferedData->size()};

      Buffer responseBuffer = client.send(buffer);
      // ResponseMessage response{responseBuffer.data()};
      // DDLResponseMessage responsePayload(response.getPayloadBuffer());
      
  }

private:
  blazingdb::protocol::Client client;
};


}
}
using namespace blazingdb::protocol;


int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
  CalciteClient client{connection};
  
  {
     std::string query = "select * from orders";
    try {
        std::string logicalPlan = client.getLogicalPlan(query);
        std::cout << logicalPlan << std::endl;
    } catch (std::exception &error) {
        std::cout << error.what() << std::endl;
    }
  }
  {
    std::string query = "cas * from orders";
    try {
        client.updateSchema(query);
        
    } catch (std::exception &error) {
        std::cout << error.what() << std::endl;
    }
  }
  return 0;
}
