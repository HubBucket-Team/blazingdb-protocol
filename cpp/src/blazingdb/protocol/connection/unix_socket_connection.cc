#include "unix_socket_connection.h"

#include <stdexcept>

#include <unistd.h>

namespace blazingdb {
namespace protocol {

UnixSocketConnection::UnixSocketConnection(const std::string &path)
    : Connection(socket(AF_UNIX, SOCK_STREAM, 0), path) {
  if (fd_ == -1) { throw std::runtime_error("socket error"); }
}

UnixSocketConnection::~UnixSocketConnection() { close(fd_); }

void UnixSocketConnection::initialize() const noexcept {}

}  // namespace protocol
}  // namespace blazingdb
