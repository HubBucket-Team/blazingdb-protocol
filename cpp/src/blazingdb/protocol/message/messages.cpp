#include <blazingdb/protocol/message/messages.h>

#include <string>
#include <functional>
#include <typeinfo>    

#include <blazingdb/protocol/api.h>
#include <iostream>
#include "flatbuffers/flatbuffers.h"
#include <blazingdb/protocol/all_generated.h>

namespace blazingdb {
namespace protocol {




  ResponseMessage::ResponseMessage (const uint8_t* buffer) 
      : IMessage()
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Response>(buffer);
      status_ = pointer->status();
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
  }

  ResponseMessage::ResponseMessage(Status status, std::shared_ptr<flatbuffers::DetachedBuffer>& buffer) : IMessage() {
      status_ = status;
      _copy_payload = buffer;

      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
  }

  ResponseMessage::ResponseMessage(Status status, IMessage& payload) : IMessage() {
    status_ = status;
    _copy_payload = payload.getBufferData();

    payloadBuffer = _copy_payload->data();
    payloadBufferSize = _copy_payload->size();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> ResponseMessage::getBufferData() const  {
    flatbuffers::FlatBufferBuilder builder{0};

    auto payload_offset = builder.CreateVector(payloadBuffer, payloadBufferSize);
    auto root_offset = CreateResponse(builder, status_, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  Status ResponseMessage::getStatus() {
    return status_;
  } 
  const uint8_t* ResponseMessage::getPayloadBuffer() {
    return payloadBuffer;
  }


  ResponseErrorMessage::ResponseErrorMessage(const std::string& error) : IMessage(), error (error)
  {
  }
  
  ResponseErrorMessage::ResponseErrorMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::ResponseError>(buffer);
    
    error = std::string{pointer->errors()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> ResponseErrorMessage::getBufferData() const   {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto string_offset = builder.CreateString(error);
    auto root_offset = CreateResponseError(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string ResponseErrorMessage::getMessage () {
    return error;
  }

static inline const Header * GetHeaderPtr (const uint8_t* buffer) {
  return flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer)->header();
}

  RequestMessage::RequestMessage (const uint8_t* buffer) 
    : IMessage(), header{GetHeaderPtr(buffer)->messageType(),
                         GetHeaderPtr(buffer)->accessToken() } 
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer);
      auto payloadBuffer = (uint8_t*)pointer->payload()->data();
      auto payloadBufferSize = pointer->payload()->size();
      this->payload = Buffer{payloadBuffer, payloadBufferSize};
  }

  RequestMessage::RequestMessage(Header &&_header, Buffer& payload) 
      : IMessage(), header{_header} 
  {
      // _copy_payload = payload.getBufferData(); 
      auto payloadBuffer = payload.data();
      auto payloadBufferSize = payload.size();

      this->payload = Buffer{payloadBuffer, payloadBufferSize};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> RequestMessage::getBufferData() const {
    flatbuffers::FlatBufferBuilder builder{0};
    auto payload_offset = builder.CreateVector(this->payload.data(), this->payload.size());
    auto root_offset = CreateRequest(builder, &header, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  Buffer RequestMessage::getPayloadBuffer() {
    return payload;
  }

  size_t RequestMessage::getPayloadBufferSize() {
    return  payload.size();
  }

  int8_t  RequestMessage::messageType() const { 
    return header.messageType();
  }

  uint64_t  RequestMessage::accessToken() const {
    return header.accessToken();
  }






  ZeroMessage::ZeroMessage()
      : IMessage()
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> ZeroMessage::getBufferData( ) const  {
    flatbuffers::FlatBufferBuilder builder{};
    auto root_offset = builder.CreateString("");
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }


 
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





} // namespace protocol
} // namespace blazingdb