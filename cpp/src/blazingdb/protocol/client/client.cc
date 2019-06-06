#include "client.h"

#include <string>
#include <stdexcept>

#include "../utilities/io_util.h"

namespace blazingdb {
namespace protocol {

Client::Client(const Connection &connection) : connection_(connection) {
  int result =
      connect(connection.fd(), connection.address(), connection.length());

  if (result == -1) { throw std::runtime_error("connect error"); }
}

Buffer Client::send(const Buffer &buffer) {
  util::write_buffer(connection_.fd(), buffer);
  
  std::cout << "escribiiooooooooooo " << buffer.size() <<std::endl;
  
  
  Buffer response_buffer;
  util::read_buffer(connection_.fd(), response_buffer);

  std::cout << "leyyyyooo decalcite " << response_buffer.size() << std::endl;

  return response_buffer;
}

Buffer Client::send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
  return this->send(Buffer{buffer->data(), buffer->size()});
}


}  // namespace protocol
}  // namespace blazingdb
