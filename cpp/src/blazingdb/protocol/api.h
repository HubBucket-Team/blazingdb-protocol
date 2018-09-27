#ifndef BLAZINGDB_PROTOCOL_API_H_
#define BLAZINGDB_PROTOCOL_API_H_

#include "buffer/buffer.h"
#include "client/client.h"
#include "connection/unix_socket_connection.h"
#include "server/server.h"
#include "messages.h"

namespace blazingdb {
namespace protocol {

// TODO(gcca): serializable through output stream
template <class Frame>
class Protocol {
public:
  //// TODO(gcca): reading input stream

  const Frame &frameFrom(const Buffer &buffer) const {
    const auto *frame = reinterpret_cast<const Frame *>(buffer.data());
    return *frame;
  }

  const Buffer payloadFrom(const Buffer &buffer) const {
    return buffer.slice(static_cast<std::ptrdiff_t>(sizeof(Frame)));
  }
};

template <class Frame>
class ProtocolRoutingServer : public Server {
  using Protocol_ = Protocol<Frame>;

public:
  explicit ProtocolRoutingServer(const Connection &connection,
                                 const Protocol_ &protocol)
      : Server(connection), protocol_(protocol) {}

  template <class Callable>
  void handle(Callable &&callback) const {
    Server::handle([this, &callback](const Buffer &buffer) -> void {
      const Frame &frame = protocol_.frameFrom(buffer);

      if (frame.kind < 0) { return; }

      const Buffer payloadBuffer = protocol_.payloadFrom(buffer);

      callback(payloadBuffer);
    });
  }

  template <class RequestFB, class Callable>
  void Register(Callable &callback) {
    callback(nullptr);
  }

private:
  const Protocol_ &protocol_;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
