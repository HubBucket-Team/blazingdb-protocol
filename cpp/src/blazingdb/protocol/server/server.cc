#include "server.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include <unistd.h>

namespace blazingdb {
namespace protocol {


Server::Server(const Connection &connection) : connection_(connection) {
  unlink(static_cast<const char *>(connection_.address()->sa_data));

  if (bind(connection_.fd(), connection_.address(), connection_.length()) ==
      -1) {
    throw std::runtime_error("bind error");
  }

  if (listen(connection_.fd(), 100) == -1) {
    throw std::runtime_error("listen error");
  }
} 

void Server::_Start(const __HandlerBaseType &handler) const {
  for (;;) {
    std::thread thread ([&]() {
      int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);
      if (fd == -1) { throw std::runtime_error("accept error"); }
      uint32_t length;
      ssize_t nread = read(fd, (void*)&length, sizeof(uint32_t));
      while (nread > 0) {
        std::uint8_t buffer[length];
        nread = read(fd, buffer, length);
        auto responseBuffer =
            handler->call(Buffer(buffer, static_cast<std::size_t>(nread)));
        uint32_t responseBufferLength = responseBuffer.size();
        ssize_t written_bytes =
            write(fd, (void*)&responseBufferLength, sizeof(uint32_t));
        written_bytes =
            write(fd, responseBuffer.data(), responseBuffer.size());
        if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
          throw std::runtime_error("write error");
        }
        nread = read(fd, (void*)&length, sizeof(uint32_t));
      }
      if (nread == -1) {
        throw std::runtime_error("error read");
      }
      if (nread == 0) {
        close(fd);
      } else {
        throw std::runtime_error("unreachable");
      }
    });
    thread.join();
  }
}

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
