#pragma once

#include <string>
#include <functional>
#include <typeinfo>    

#include <blazingdb/protocol/api.h>
#include <iostream>
#include "flatbuffers/flatbuffers.h"
#include "generated/all_generated.h"

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


template<typename SchemaType>
class StringTypeMessage : public IMessage {
public: 
  StringTypeMessage(const std::string& string) : IMessage(), string_value (string){

  }
  
  using PointerToMethod = const flatbuffers::String* (SchemaType::*)() const;

  StringTypeMessage (const uint8_t* buffer, PointerToMethod pmfn)
    : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<SchemaType>(buffer);
    // auto string_buffer = std::invoke(pointer, pmfn);
    auto string_buffer = (pointer->*pmfn)();
    string_value = std::string {string_buffer->c_str()};
  }

  template<class CreateFunctionPtr>
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferDataUsing(CreateFunctionPtr &&create_function) const  {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto root_offset = create_function(builder, string_value.c_str());
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  
protected:
  std::string string_value;
 };


auto MakeRequest(int8_t message_header, IMessage&& payload) -> std::shared_ptr<flatbuffers::DetachedBuffer>{
  RequestMessage request{message_header, payload}; 
  auto bufferedData = request.getBufferData();
  return bufferedData;
}

template <typename ResponseType>
ResponseType MakeResponse (Buffer responseBuffer) {
  ResponseMessage response{responseBuffer.data()};
  if (response.getStatus() == Status_Error) {
    ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
    throw std::runtime_error(errorMessage.getMessage());
  }
  ResponseType responsePayload(response.getPayloadBuffer());
  return responsePayload;
}



// template <typename ResponseType>
// ResponseType MakeResponse (int8_t status, IMessage&& payload) {
//   ResponseMessage response{responseBuffer.data()};
//     if (response.getStatus() == Status_Error) {
//       ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
//       throw std::runtime_error(errorMessage.getMessage());
//     }
//     ResponseType responsePayload(response.getPayloadBuffer());
//     return responsePayload;
// }

}
}