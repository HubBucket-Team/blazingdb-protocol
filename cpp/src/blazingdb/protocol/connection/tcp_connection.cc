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
}

TCPConnection::~TCPConnection() { close(fd_); }

void TCPConnection::initialize() const noexcept {
  struct hostent *hostent_ = gethostbyname(ip_.c_str());
  if (nullptr == hostent_) {
    // TODO: handle error
  }

  bcopy(hostent_->h_addr,
        reinterpret_cast<void *>(
            const_cast<TCPConnection *>(this)->addr_.sin_addr.s_addr),
        static_cast<std::size_t>(hostent_->h_length));
}

}  // namespace protocol
}  // namespace blazingdb
