#include "client.h"

#include <unistd.h>

#include <string>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <thread>

#include "../connection/tcp_connection.h"

#include "../utilities/io_util.h"

namespace blazingdb {
namespace protocol {

Client::Client(const Connection &connection) : connection_(connection) {

#ifdef USE_UNIX_SOCKETS

  int result = connect(connection.fd(), connection.address(), connection.length());

#else

  // TCP

  unlink(static_cast<const char *>(connection_.address()->sa_data));

  if (bind(connection_.fd(), connection_.address(), connection_.length()) == -1) {
    throw std::runtime_error("TCP client: bind error");
  }

  // Connect to the remote server
  struct sockaddr_in remoteaddr;
  remoteaddr.sin_family = AF_INET;
  remoteaddr.sin_addr.s_addr = inet_addr(const_cast<TCPConnection>(connection_).ip_.c_str());
  remoteaddr.sin_port = htons(server_port);

  int result = connect(connection.fd(), (struct sockaddr *)&remoteaddr, sizeof(remoteaddr));

#endif

  if (result == -1) { throw std::runtime_error("connect error"); }
}

Buffer Client::send(const Buffer &buffer) {
  util::write_buffer(connection_.fd(), buffer);
  
  std::cout << "escribiiooooooooooo " << buffer.size() <<std::endl;
  
  
  Buffer response_buffer;
  util::read_buffer(connection_.fd(), response_buffer);

  std::cout << "leyyyyooo decalcite " << response_buffer.size() << std::endl;

  return response_buffer;
}

Buffer Client::send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
  return this->send(Buffer{buffer->data(), buffer->size()});
}


}  // namespace protocol
}  // namespace blazingdb
