#include "server.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cstdint>
#include <stdexcept>

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
    int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);
    if (fd == -1) { throw std::runtime_error("accept error"); }
    StackBuffer buffer;
    int nread = read(fd, buffer.data(), buffer.size());

    // auto header_size = sizeof (Header); //>??
    // auto buffer_header = read(fd, buffer.data(), header_size );
    // auto payloadLength = flatbuffers::GetRoot<Header>(buffer_header)->payloadLength;
    // int nread = read(fd, buffer.data() + header_size, payloadLength);

    //@todo: check this function for recovering byte per byte
    // ssize_t nread = _GetRequest(fd, buffer);
    if (nread > 0) {
      Buffer responseBuffer =
          handler->call(Buffer(buffer.data(), static_cast<std::size_t>(nread)));
      ssize_t written_bytes =
          write(fd, responseBuffer.data(), responseBuffer.size());
      if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
        throw std::runtime_error("write error");
      }
    }
    else if (nread == -1) {
      throw std::runtime_error("error read");
    } else {
      throw std::runtime_error("unreachable");
    }
    close(fd);
  }
}

}  // namespace protocol
}  // namespace blazingdb
