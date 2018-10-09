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

  ResponseMessage(Status status, std::shared_ptr<flatbuffers::DetachedBuffer>& buffer) : IMessage() {
      status_ = status;
      _copy_payload = buffer;

      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
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

static inline const Header * GetHeaderPtr (const uint8_t* buffer) {
  return flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer)->header();
}

class RequestMessage : public IMessage {
public:  
  RequestMessage (const uint8_t* buffer) 
    : IMessage(), header{GetHeaderPtr(buffer)->messageType(),
                         GetHeaderPtr(buffer)->payloadLength(), 
                         GetHeaderPtr(buffer)->accessToken() } 
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer);
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
      
  }
  RequestMessage(Header &&_header, IMessage& payload) 
      : IMessage(), header{_header} 
  {
      _copy_payload = payload.getBufferData(); 
      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto payload_offset = builder.CreateVector(payloadBuffer, payloadBufferSize);
    auto root_offset = CreateRequest(builder, &header, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  Buffer getPayloadBuffer() {
    return Buffer{payloadBuffer, payloadBufferSize};
  }

  size_t getPayloadBufferSize() {
    return  payloadBufferSize;
  }

  int8_t  messageType() const { 
    return header.messageType();
  }

  uint64_t  accessToken() const {
    return header.accessToken();
  }


private:
    Header            header;
    const uint8_t*    payloadBuffer;
    size_t            payloadBufferSize;
    std::shared_ptr<flatbuffers::DetachedBuffer>  _copy_payload; 
};


template<typename T, typename SchemaType>
class TypedMessage : public IMessage {
public:
  TypedMessage(const T& val) : IMessage(), value_ (val){

  }

  using PointerToMethod = T (SchemaType::*)() const;

  TypedMessage (const uint8_t* buffer, PointerToMethod pmfn)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<SchemaType>(buffer);
    value_ = (pointer->*pmfn)();
  }

  template<class CreateFunctionPtr>
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferDataUsing(CreateFunctionPtr &&create_function) const  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto root_offset = create_function(builder, value_);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

protected:
  T value_;
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


//@todo create ZeroMessage Schema
class  ZeroMessage : public StringTypeMessage<orchestrator::DMLResponse> {
public:

  ZeroMessage()
      : StringTypeMessage<orchestrator::DMLResponse>("")
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(orchestrator::CreateDMLRequestDirect);
  }
};



auto MakeRequest(int8_t message_type, uint64_t payloadLength, uint64_t sessionToken, IMessage&& payload) -> std::shared_ptr<flatbuffers::DetachedBuffer>{
  RequestMessage request{ Header{message_type, payloadLength, sessionToken}, payload}; 
  auto bufferedData = request.getBufferData();
  return bufferedData;
}
auto MakeRequest(int8_t message_type, uint64_t payloadLength, uint64_t sessionToken, IMessage& payload) -> std::shared_ptr<flatbuffers::DetachedBuffer>{
  RequestMessage request{ Header{message_type, payloadLength, sessionToken}, payload}; 
  auto bufferedData = request.getBufferData();
  return bufferedData;
}
 

template <typename ResponseType>
ResponseType MakeResponse (Buffer &responseBuffer) {
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