#include "client.h"

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

Client::Client(const Connection &connection) : connection_(connection) {
  int result =
      connect(connection.fd(), connection.address(), connection.length());

  if (result == -1) { throw std::runtime_error("connect error"); }
}

Buffer Client::send(const Buffer &buffer) {
  ssize_t written_bytes = write(connection_.fd(), buffer.data(), buffer.size());

  if (static_cast<std::size_t>(written_bytes) != buffer.size()) {
    throw std::runtime_error("write error");
  }

  StackBuffer responseBuffer;
  ssize_t nread =
      read(connection_.fd(), responseBuffer.data(), responseBuffer.size());

  if (nread == -1) { throw std::runtime_error("error read"); }

  return responseBuffer;
}

Buffer Client::send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
  return this->send(Buffer{buffer->data(), buffer->size()});
}

}  // namespace protocol
}  // namespace blazingdb
