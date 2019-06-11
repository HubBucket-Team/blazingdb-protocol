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
  if (bind(connection_.fd(), connection_.address(), connection_.length()) == -1) {
    throw std::runtime_error("Server: bind error");
  }

  if (listen(connection_.fd(), 100) == -1) {
    throw std::runtime_error("Server: listen error");
  }
}

void Server::_Start(const __HandlerBaseType &handler) const {
  for (;;) {
    struct sockaddr_in client_address;
    socklen_t client_address_size = sizeof(client_address);
      
    int fd = accept4(connection_.fd(), (struct sockaddr*)&client_address, &client_address_size, SOCK_CLOEXEC);
    
    if (fd == -1) { throw std::runtime_error("accept error"); }

    const std::string client_ip = inet_ntoa(client_address.sin_addr);
    
    std::cout << "Server is handling the response for client: " << client_ip << std::endl;
    
    Buffer temp_buffer;
    util::read_buffer(fd, temp_buffer);

    Buffer response_buffer = handler->call(temp_buffer);

    util::write_buffer(fd, response_buffer);

    close(fd);    
  }
}

}  // namespace protocol
}  // namespace blazingdb
