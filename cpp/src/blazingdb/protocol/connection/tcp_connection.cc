#include "tcp_connection.h"

#include <stdexcept>

#include <cstdio>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

namespace blazingdb {
namespace protocol {

TCPConnection::TCPConnection(const std::string &ip, const std::string &port)
    : Connection(socket(AF_INET, SOCK_STREAM, 0), port), ip_{ip} {
  if (fd_ == -1) { throw std::runtime_error("socket error"); }
  int on = 1;
  if (setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
    throw std::runtime_error("set socket option error");
  }
}

TCPConnection::~TCPConnection() { close(fd_); }

void TCPConnection::initialize() const noexcept {
  const_cast<TCPConnection *>(this)->addr_.sin_addr.s_addr =
      inet_addr(ip_.c_str());
}

}  // namespace protocol
}  // namespace blazingdb
