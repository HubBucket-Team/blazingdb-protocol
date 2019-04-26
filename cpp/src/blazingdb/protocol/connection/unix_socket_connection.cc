#ifdef USE_UNIX_SOCKETS

#include "unix_socket_connection.h"

#include <stdexcept>

#include <unistd.h>
#include <sys/un.h>

namespace blazingdb {
namespace protocol {

UnixSocketConnection::UnixSocketConnection(const ConnectionAddress &connectionAddress)
    : Connection(socket(AF_UNIX, SOCK_STREAM, 0), connectionAddress.unix_socket_path) {
  if (fd_ == -1) { throw std::runtime_error("socket error"); }
}

UnixSocketConnection::~UnixSocketConnection() { close(fd_); }

void UnixSocketConnection::initialize() const noexcept {}

}  // namespace protocol
}  // namespace blazingdb

#endif
