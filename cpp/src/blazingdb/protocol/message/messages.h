#pragma once

#include <string>
#include <functional>
#include <typeinfo>    

#include <blazingdb/protocol/api.h>
#include <iostream>
#include "flatbuffers/flatbuffers.h"
#include <blazingdb/protocol/all_generated.h>

namespace blazingdb {
namespace protocol {

struct IMessage {

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
    flatbuffers::FlatBufferBuilder builder{0};

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
                         GetHeaderPtr(buffer)->accessToken() } 
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer);
      auto payloadBuffer = (uint8_t*)pointer->payload()->data();
      auto payloadBufferSize = pointer->payload()->size();
      this->payload = Buffer{payloadBuffer, payloadBufferSize};
  }

  RequestMessage(Header &&_header, Buffer& payload) 
      : IMessage(), header{_header} 
  {
      // _copy_payload = payload.getBufferData(); 
      auto payloadBuffer = payload.data();
      auto payloadBufferSize = payload.size();

      this->payload = Buffer{payloadBuffer, payloadBufferSize};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{0};
    auto payload_offset = builder.CreateVector(this->payload.data(), this->payload.size());
    auto root_offset = CreateRequest(builder, &header, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  Buffer getPayloadBuffer() {
    return payload;
  }

  size_t getPayloadBufferSize() {
    return  payload.size();
  }

  int8_t  messageType() const { 
    return header.messageType();
  }

  uint64_t  accessToken() const {
    return header.accessToken();
  }

private:
    Header            header;
    Buffer            payload;
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
    flatbuffers::FlatBufferBuilder builder{0};
    auto root_offset = create_function(builder, string_value.c_str());
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  
protected:
  std::string string_value;
 };


class  ZeroMessage : public IMessage {
public:

  ZeroMessage()
      : IMessage()
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{};
    auto root_offset = builder.CreateString("");
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
};

 
auto MakeRequest(int8_t message_type, uint64_t sessionToken, Buffer& payload) -> Buffer {
  RequestMessage request{ Header{message_type, sessionToken}, payload}; 
  auto bufferedData = request.getBufferData();
  return Buffer{bufferedData->data(), bufferedData->size()};
} 

 
auto MakeRequest(int8_t message_type, uint64_t sessionToken, IMessage& message) -> Buffer {
  auto payload = Buffer{message.getBufferData()};
  RequestMessage request(Header{message_type, sessionToken}, payload); 
  auto bufferedData = request.getBufferData();
  return Buffer{bufferedData->data(), bufferedData->size()};
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

} // namespace protocol
} // namespace blazingdb