#ifndef BLAZINGDB_PROTOCOL_SERVER_SERVER_H_
#define BLAZINGDB_PROTOCOL_SERVER_SERVER_H_
#include <zmq.h>

#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../buffer/buffer.h"
#include "../connection/connection.h"

namespace blazingdb {
namespace protocol {
 
class Server {
public:
  explicit Server(const Connection &connection);

  template <class Callable>
  void handle(Callable &&callback) const {
    _Start(_MakeHandler(callback));
  }

private:
  const Connection &connection_;

  struct _HandlerBase;
  using __HandlerBaseType = std::shared_ptr<_HandlerBase>;

  void _Start[[noreturn]](const __HandlerBaseType &) const;

  struct _HandlerBase {
    inline virtual ~_HandlerBase() = default;
    virtual Buffer call(const Buffer &) = 0;
  };

  template <class Callable>
  class _Handler : public _HandlerBase {
  public:
    _Handler(Callable &&callback)
        : callback_(std::forward<Callable>(callback)) {}

    Buffer call(const Buffer &buffer) { return callback_(buffer); }

  private:
    Callable callback_;
  };

  template <class Callable>
  std::shared_ptr<_Handler<Callable>> _MakeHandler(Callable &&callback) const {
    return std::make_shared<_Handler<Callable>>(
        std::forward<Callable>(callback));
  }
};


class ZeroMqServer {
public:
  explicit ZeroMqServer(const std::string &connection) {
    context = zmq_ctx_new();
    socket = zmq_socket (context, ZMQ_REP);
    auto rc = zmq_bind(socket, connection.c_str());
    //assert (rc == 0);
  }

  using Callable = blazingdb::protocol::Buffer (*)(const blazingdb::protocol::Buffer &requestBuffer);
  void handle(Callable &&callback)  {
    zmq_msg_t msg;
    int rc = zmq_msg_init(&msg);
    // assert(rc != 0);
    zmq_msg_recv(&msg, socket, 0);
    auto size = zmq_msg_size(&msg);
    Buffer responseBuffer((uint8_t*) zmq_msg_data(&msg), size);
    Buffer bufferedData = callback(responseBuffer);
    zmq_send (socket, bufferedData.data(), bufferedData.size(), 0);
    zmq_msg_close (&msg);
  }
private:
    void *context;
    void * socket;
};
  


}  // namespace protocol
}  // namespace blazingdb

#endif
