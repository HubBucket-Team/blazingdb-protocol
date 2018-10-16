#include <iostream>
#include <string>

#include <blazingdb/protocol/api.h>

int main() {
   blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
   blazingdb::protocol::Client client(connection);

   blazingdb::protocol::Buffer buffer(
       reinterpret_cast<const std::uint8_t*>("BlazingDB Request"), 17);

   blazingdb::protocol::Buffer responseBuffer = client.send(buffer);
   std::cout << responseBuffer.data() << std::endl;

  blazingdb::protocol::Buffer buffer2(
      reinterpret_cast<const std::uint8_t*>("BlazingDB Request 2"), 19);

  blazingdb::protocol::Buffer responseBuffer2 = client.send(buffer2);
  std::cout << responseBuffer2.data() << std::endl;

  return 0;
}
