
#include <string>
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
      status_ = pointer->status();
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
  }
  ResponseMessage(Status status, IMessage& payload) : IMessage() {
      status_ = status;
      _copy_payload = payload.getBufferData();

      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    flatbuffers::FlatBufferBuilder builder{1024};

    auto payload_offset = builder.CreateVector(payloadBuffer, payloadBufferSize);
    auto root_offset = CreateResponse(builder, status_, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  Status getStatus() {
    return status_;
  } 
  const uint8_t* getPayloadBuffer() {
    return payloadBuffer;
  }
private:
    Status            status_;
    uint8_t*          payloadBuffer;
    size_t            payloadBufferSize;
    std::shared_ptr<flatbuffers::DetachedBuffer>  _copy_payload; 

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

class ResponseErrorMessage : public IMessage {
public:  

  ResponseErrorMessage(const std::string& error) : IMessage(), error (error)
  {
  }
  
  ResponseErrorMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::ResponseError>(buffer);
    
    error = std::string{pointer->errors()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto string_offset = builder.CreateString(error);
    auto root_offset = CreateResponseError(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getMessage () {
    return error;
  }
  
private:
  std::string error;
};


class RequestMessage : public IMessage {
public:  
  RequestMessage (const uint8_t* buffer) : IMessage() {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer);
      headerType = pointer->header();
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
      
  }
  RequestMessage(int8_t header, IMessage& payload) : IMessage() {
      headerType = header;
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

  int8_t header() { 
    return headerType;
  }
   

private:
    int8_t            headerType;
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

  std::string getQuery() const {
    return query;
  }

private:
  std::string query;
};



class DDLRequestMessage : public IMessage {
public: 

  DDLRequestMessage(const std::string& query) : IMessage(), query (query){
  }
  
  DDLRequestMessage (const uint8_t* buffer) : IMessage() {
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

  std::string getQuery() const {
    return query;
  }

private:
  std::string query;
};

}
}