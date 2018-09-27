#include "server.h"

#include <cstdint>
#include <stdexcept>

#include <unistd.h>

namespace blazingdb {
namespace protocol {

namespace {

class StackBuffer : public Buffer {
public:
  static const std::size_t MAX_SIZE = 4096;

  StackBuffer()
      : Buffer(static_cast<const std::uint8_t *const>(actual_data_), MAX_SIZE),
        actual_data_{0} {}

  std::uint8_t *data() { return static_cast<std::uint8_t *>(actual_data_); }

private:
  std::uint8_t actual_data_[MAX_SIZE];
};

}  // namespace

Server::Server(const Connection &connection) : connection_(connection) {
  unlink(static_cast<const char *>(connection_.address()->sa_data));

  if (bind(connection_.fd(), connection_.address(), connection_.length()) ==
      -1) {
    throw std::runtime_error("bind error");
  }

  if (listen(connection_.fd(), 1) == -1) {
    throw std::runtime_error("listen error");
  }
}

void Server::_Start(const __HandlerBaseType &handler) const {
  for (;;) {
    int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);

    if (fd == -1) { throw std::runtime_error("accept error"); }

    StackBuffer buffer;
    ssize_t nread = read(fd, buffer.data(), buffer.size());

    if (nread > 0) {
      Buffer responseBuffer =
          handler->call(Buffer(buffer.data(), static_cast<std::size_t>(nread)));

      ssize_t written_bytes =
          write(fd, responseBuffer.data(), responseBuffer.size());

      if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
        throw std::runtime_error("write error");
      }
    } else if (nread == -1) {
      throw std::runtime_error("error read");
    } else if (nread == 0) {
      close(fd);
    } else {
      throw std::runtime_error("unreachable");
    }
  }
}

}  // namespace protocol
}  // namespace blazingdb
