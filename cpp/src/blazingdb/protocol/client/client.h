#ifndef BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_
#define BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_

#include <memory>
#include "../buffer/buffer.h"
#include "../connection/connection.h"
#include "flatbuffers/flatbuffers.h"

namespace blazingdb {
namespace protocol {

class Client {
public:
  explicit Client(const Connection &connection);

  Buffer send(const Buffer &buffer);

  Buffer send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer);

private:
  const Connection &connection_;
};

class ZeroMqClient {
public:
  explicit ZeroMqClient(const std::string &connection);
 
 ~ZeroMqClient(); 

  Buffer send(const Buffer &buffer);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};


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

#endif
