#include "client.h"

#include <string>
#include <stdexcept>

#include <unistd.h>

namespace blazingdb {
namespace protocol {

Client::Client(const Connection &connection) : connection_(connection) {
  int result =
      connect(connection.fd(), connection.address(), connection.length());

  if (result == -1) { throw std::runtime_error("connect error"); }
}

Buffer Client::send(const Buffer &buffer) {
  int bufferLength = buffer.size();
  ssize_t written_bytes = write(connection_.fd(), (void*)&bufferLength, sizeof(int));
  written_bytes = write(connection_.fd(), (void*)buffer.data(), bufferLength);
  if (static_cast<std::size_t>(written_bytes) != buffer.size()) {
    throw std::runtime_error("write error");
  }

  uint32_t responseBufferLength;
  ssize_t nread =
      read(connection_.fd(), (void*)&responseBufferLength, sizeof(uint32_t));
  if (nread == -1) { throw std::runtime_error("error read buffer length in Client::send"); }

  Buffer responseBuffer;
  responseBuffer.resize(responseBufferLength);
  nread =
      read(connection_.fd(), (void*)responseBuffer.data(), responseBufferLength);

  if (nread == -1) { throw std::runtime_error("error read in Client::send"); }

  return responseBuffer;
}

Buffer Client::send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
  return this->send(Buffer{buffer->data(), buffer->size()});
}


}  // namespace protocol
}  // namespace blazingdb
