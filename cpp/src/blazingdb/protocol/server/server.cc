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

ssize_t Server::_GetRequest(int fd, StackBuffer &buffer) const {
  std::basic_string<uint8_t> request;
  while (true) {
    uint8_t next_byte;
    int n = recv(fd, &next_byte, 1, 0);
    if (n <= 0) {
      break;
    }
    request.append(&next_byte, 1);
    if (next_byte == (uint8_t)'\n') {
      break;
    }
  }
  assert (request.size() <= buffer.size());
  std::copy(request.c_str(), request.c_str() + request.size(), buffer.data());
  return request.size();
}

void Server::_Start(const __HandlerBaseType &handler) const {
  for (;;) {
    std::thread thread ([&]() {
      int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);
      if (fd == -1) { throw std::runtime_error("accept error"); }
      uint32_t length;
      ssize_t nread = read(fd, (void*)&length, sizeof(uint32_t));
      while (nread > 0) {
        std::uint8_t buffer[length];
        nread = read(fd, buffer, length);
        auto responseBuffer =
            handler->call(Buffer(buffer, static_cast<std::size_t>(nread)));
        uint32_t responseBufferLength = responseBuffer.size();
        ssize_t written_bytes =
            write(fd, (void*)&responseBufferLength, sizeof(uint32_t));
        written_bytes =
            write(fd, responseBuffer.data(), responseBuffer.size());
        if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
          throw std::runtime_error("write error");
        }
        nread = read(fd, (void*)&length, sizeof(uint32_t));
      }
      if (nread == -1) {
        throw std::runtime_error("error read");
      }
      if (nread == 0) {
        close(fd);
      } else {
        throw std::runtime_error("unreachable");
      }
    });
    thread.join();
  }
}

}  // namespace protocol
}  // namespace blazingdb
