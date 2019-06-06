#ifndef USE_UNIX_SOCKETS

#include "tcp_connection.h"

#include <stdexcept>

#include <cstdio>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

namespace blazingdb {
namespace protocol {

TCPConnection::TCPConnection(const ConnectionAddress &connectionAddress)
    : Connection(socket(AF_INET, SOCK_STREAM, 0), connectionAddress.tcp_port), ip_{connectionAddress.tcp_host} {
  if (fd_ == -1) { throw std::runtime_error("socket error"); }
  int on = 1;
  if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
    throw std::runtime_error("set socket option error");
  }
  
  this->initialize();

  in_addr_t ip = addr_.sin_addr.s_addr;
  std::string socketAddress(std::to_string(ip));

}

TCPConnection::~TCPConnection() { close(fd_); }

void TCPConnection::initialize() const noexcept {
}

}  // namespace protocol
}  // namespace blazingdb

#endif
