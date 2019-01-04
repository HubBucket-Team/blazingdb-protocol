#include "client.h"

#include <string>
#include <stdexcept>

#include <unistd.h>

namespace blazingdb {
namespace protocol {
 
Client::Client(const Connection &connection) : connection_(connection) {
  int result =
      connect(connection.fd(), connection.address(), connection.length());

  if (result == -1) { throw std::runtime_error("connect error"); }
}

Buffer Client::send(const Buffer &buffer) {
  int bufferLength = buffer.size();
  ssize_t written_bytes = write(connection_.fd(), (void*)&bufferLength, sizeof(int));
  written_bytes = write(connection_.fd(), (void*)buffer.data(), bufferLength);
  if (static_cast<std::size_t>(written_bytes) != buffer.size()) {
    throw std::runtime_error("write error");
  }

  uint32_t responseBufferLength;
  ssize_t nread =
      read(connection_.fd(), (void*)&responseBufferLength, sizeof(uint32_t));
  
  Buffer responseBuffer;
  responseBuffer.resize(responseBufferLength);
  nread =
      read(connection_.fd(), (void*)responseBuffer.data(), responseBufferLength);

  if (nread == -1) { throw std::runtime_error("error read"); }

  return responseBuffer;
}

Buffer Client::send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
  return this->send(Buffer{buffer->data(), buffer->size()});
}


#include <zmq.h>
  
class ZeroMqClient::impl {
public:
  impl(const std::string &connection): context{zmq_ctx_new()}, socket{ zmq_socket (context, ZMQ_REQ) } {
    auto rc = zmq_connect(socket, connection.c_str());
    assert (rc == 0);
 }

 Buffer send(const Buffer &bufferedData) {
    zmq_send(socket, bufferedData.data(), bufferedData.size(), 0);
    zmq_msg_t msg;
    int rc = zmq_msg_init(&msg);
    assert(rc != 0);
    zmq_msg_recv(&msg, socket, 0);
    auto size = zmq_msg_size(const_cast<zmq_msg_t *>(&msg));
    Buffer responseBuffer((uint8_t*)&msg, size);
    return responseBuffer;
  }
private:
    void *context;
    void * socket;
};


ZeroMqClient::ZeroMqClient(const std::string &connection) :  pimpl{std::make_unique<impl>(connection)}{
}

ZeroMqClient::~ZeroMqClient()
{}

Buffer ZeroMqClient::send(const Buffer &buffer) {
  this->pimpl->send(buffer);
}

}  // namespace protocol
}  // namespace blazingdb
