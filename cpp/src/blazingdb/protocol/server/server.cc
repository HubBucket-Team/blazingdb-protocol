#include "server.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include "../utilities/io_util.h"

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

    Buffer temp_buffer;
    util::read_buffer(fd, temp_buffer);

    Buffer response_buffer = handler->call(temp_buffer);

    util::write_buffer(fd, response_buffer);

    close(fd);    
  }
}


}  // namespace protocol
}  // namespace blazingdb
