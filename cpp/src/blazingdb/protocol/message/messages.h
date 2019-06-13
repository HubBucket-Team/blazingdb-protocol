#ifndef BLAZINGDB_PROTOCOL_MESSAGE_MESSAGES_H
#define BLAZINGDB_PROTOCOL_MESSAGE_MESSAGES_H

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
  ResponseMessage (const uint8_t* buffer);

  ResponseMessage(Status status, std::shared_ptr<flatbuffers::DetachedBuffer>& buffer);

  ResponseMessage(Status status, IMessage& payload);

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override;
  Status getStatus();
  const uint8_t* getPayloadBuffer();
private:
    Status            status_;
    uint8_t*          payloadBuffer;
    size_t            payloadBufferSize;
    std::shared_ptr<flatbuffers::DetachedBuffer>  _copy_payload; 

 };

class ResponseErrorMessage : public IMessage {
public:  

  ResponseErrorMessage(const std::string& error);
  
  ResponseErrorMessage (const uint8_t* buffer);

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override;

  std::string getMessage ();
  
private:
  std::string error;
};

static inline const Header * GetHeaderPtr (const uint8_t* buffer);

class RequestMessage : public IMessage {
public:  
  RequestMessage (const uint8_t* buffer);

  RequestMessage(Header &&_header, Buffer& payload);

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override;

  Buffer getPayloadBuffer();

  size_t getPayloadBufferSize();

  int8_t  messageType() const;

  uint64_t  accessToken() const;

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

  ZeroMessage();

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override;
};

 
auto MakeRequest(int8_t message_type, uint64_t sessionToken, Buffer& payload) -> Buffer;
 
auto MakeRequest(int8_t message_type, uint64_t sessionToken, IMessage& message) -> Buffer;

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

#endif  // BLAZINGDB_PROTOCOL_MESSAGE_MESSAGES_H