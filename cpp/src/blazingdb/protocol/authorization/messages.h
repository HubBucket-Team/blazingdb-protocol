#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"

namespace blazingdb {
namespace protocol {
namespace authorization {

class AuthRequestMessage : IMessage {
public:
  AuthRequestMessage( )
      : IMessage{}
  {
  }

  AuthRequestMessage (const uint8_t* buffer)
      : IMessage{}
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    return nullptr;
  }
};




class AuthResponseMessage : public IMessage {
public:

  AuthResponseMessage(int64_t access_token)
      : IMessage{}, access_token_{access_token}
  {
  }

  AuthResponseMessage ( AuthResponseMessage && ) = default;

  AuthResponseMessage (const uint8_t* buffer)
      :   IMessage{}
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::authorization::AuthResponse>(buffer);

    access_token_ = pointer->accessToken();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto root_offset = CreateAuthResponse(builder, access_token_);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  int64_t getAccessToken () {
    return access_token_;
  }

private:
  int64_t access_token_;
};


} // interpreter
} // protocol
} // blazingdb
