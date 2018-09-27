#include <iostream>

#include <blazingdb/protocol/api.h>

int main() {
  blazingdb::protocol::UnixSocketConnection connection(
      {"/tmp/socket", std::allocator<char>()});
  blazingdb::protocol::Server server(connection);

  server.handle([](const blazingdb::protocol::Buffer &buffer)
                    -> blazingdb::protocol::Buffer {
    std::cout << buffer.data() << std::endl;
    return blazingdb::protocol::Buffer(
        reinterpret_cast<const std::uint8_t *>("BlazingDB Response"), 18);
  });

  return 0;
}
