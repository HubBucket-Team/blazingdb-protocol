#include <protocol/api.h>
#include <protocol/config.h>

#include "flatbuffers/flatbuffers.h"
#include "QueryMessage_generated.h"

namespace blazingdb {
namespace protocol {

class BlazingQueryMessage : public IMessage {
public: 
  BlazingQueryMessage() = default;

  ~BlazingQueryMessage() = default;

  BlazingQueryMessage(std::string statement, std::string authorization)
    : IMessage{},
      statement_{std::move(statement)},
      authorization_{std::move(authorization)} {}

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    std::lock_guard<std::mutex> lock{mutex_};
    flatbuffers::FlatBufferBuilder builder{1024};

    auto statement_string_data = builder.CreateString(statement_);
    auto authorization_string_data = builder.CreateString(authorization_);
    QueryMessageBuilder tmp_builder{builder};
    tmp_builder.add_statement(statement_string_data);
    tmp_builder.add_authorization(authorization_string_data);

    FinishQueryMessageBuffer(builder, tmp_builder.Finish());

    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

private:
  std::string statement_{""};
  std::string authorization_{""};

};

}
}

using namespace blazingdb::protocol;

int main() {
  UnixSocketConnection connection("/tmp/socket");
  Client client(connection);

  const std::string statement = "select * from orders";
  const std::string authorization = PERMISSIONS_DELIM;

  BlazingQueryMessage query_message(statement, authorization);


  // Buffer buffer(
  //     reinterpret_cast<const std::uint8_t*>("BlazingDB PROTOCOL"), 19);

  auto buffer = query_message.getBufferData();

  client.send(buffer);

  return 0;
}