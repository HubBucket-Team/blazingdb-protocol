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

// This assumes buffer is at least x bytes long,
// and that the socket is blocking.
void ReadXBytes(int socket, unsigned int x, void* buffer)
{
    int bytesRead = 0;
    int result;
    while (bytesRead < x)
    {
        result = read(socket, buffer + bytesRead, x - bytesRead);
        if (result < 1 )
        {
            // Throw your error.
        }

        bytesRead += result;
    }
}

void Server::_Start(const __HandlerBaseType &handler) const {

#ifdef USE_UNIX_SOCKETS

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

#else
  
  for (;;) {
    int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);

    if (fd == -1) { throw std::runtime_error("accept error"); }

    unsigned int length = 0;
    char* buffer = 0;
    // we assume that sizeof(length) will return 4 here.
    ReadXBytes(fd, sizeof(length), (void*)(&length));
    buffer = new char[length];
    ReadXBytes(fd, length, (void*)buffer);
    
    auto responseBuffer =
        handler->call(Buffer(buffer, static_cast<std::size_t>(length)));

    uint32_t responseBufferLength = responseBuffer.size();

    ssize_t written_bytes =
        write(fd, (void *) &responseBufferLength, sizeof(uint32_t));
    written_bytes = write(fd, responseBuffer.data(), responseBuffer.size());

    if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
      throw std::runtime_error("write error");
    }

    if (nread == -1) { throw std::runtime_error("error read"); }
    close(fd);
    
    delete [] buffer;
  }

#endif

}


}  // namespace protocol
}  // namespace blazingdb
