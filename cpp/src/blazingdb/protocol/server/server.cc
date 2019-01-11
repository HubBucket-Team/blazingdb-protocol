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
    int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);

    if (fd == -1) { throw std::runtime_error("accept error"); }

    uint32_t length;
    ssize_t  nread = read(fd, (void *) &length, sizeof(uint32_t));

    std::uint8_t buffer[length];
    nread = read(fd, buffer, length);
    auto responseBuffer =
        handler->call(Buffer(buffer, static_cast<std::size_t>(nread)));

    uint32_t responseBufferLength = responseBuffer.size();

    ssize_t written_bytes =
        write(fd, (void *) &responseBufferLength, sizeof(uint32_t));
    written_bytes = write(fd, responseBuffer.data(), responseBuffer.size());

    if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
      throw std::runtime_error("write error");
    }

    if (nread == -1) { throw std::runtime_error("error read"); }
    close(fd);
  }
}


}  // namespace protocol
}  // namespace blazingdb
