#include <iostream>

#include <blazingdb/protocol/api.h>

int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
  blazingdb::protocol::Client client(connection);

  blazingdb::protocol::Buffer buffer(
      reinterpret_cast<const std::uint8_t*>("BlazingDB Request"), 17);

  blazingdb::protocol::Buffer responseBuffer = client.send(buffer);

  std::cout << responseBuffer.data() << std::endl;

  return 0;
}
