#ifndef BLAZINGDB_PROTOCOL_SERVER_SERVER_H_
#define BLAZINGDB_PROTOCOL_SERVER_SERVER_H_

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
  explicit ZeroMqServer(const std::string &connection);

  ~ZeroMqServer(); 

  using Callable = blazingdb::protocol::Buffer (*)(const blazingdb::protocol::Buffer &requestBuffer);

  void handle(Callable &&callback) ;

private:
    class impl;
    std::unique_ptr<impl> pimpl;
};


#include <zmq.h>

class ZeroMqServer::impl {
public:
  impl(const std::string &connection): context{zmq_ctx_new()}, socket{ zmq_socket (context, ZMQ_REQ) } {
    auto rc = zmq_bind(socket, connection.c_str());
    assert (rc == 0);
  }

  using Callable = blazingdb::protocol::Buffer (*)(const blazingdb::protocol::Buffer &requestBuffer);
  void handle(Callable &&callback)  {
    zmq_msg_t msg;
    int rc = zmq_msg_init(&msg);
    assert(rc != 0);
    zmq_msg_recv(&msg, socket, 0);
    auto size = zmq_msg_size(const_cast<zmq_msg_t *>(&msg));
    Buffer responseBuffer((uint8_t*)&msg, size);
    Buffer bufferedData = callback(responseBuffer);
    zmq_send (socket, bufferedData.data(), bufferedData.size(), 0);
  }
private:
    void *context;
    void * socket;
};

ZeroMqServer::ZeroMqServer(const std::string &connection) :  pimpl{std::make_unique<impl>(connection)}{
}

ZeroMqServer::~ZeroMqServer() 
{}

void ZeroMqServer::handle(ZeroMqServer::Callable &&callback)  {
  this->pimpl->handle(std::move(callback));
}


}  // namespace protocol
}  // namespace blazingdb

#endif
