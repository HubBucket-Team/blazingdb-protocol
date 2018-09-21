#include <iostream>

#include <blazingdb/protocol/api.h>

int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
  blazingdb::protocol::Server server(connection);

  server.handle([](const blazingdb::protocol::Buffer &buffer) {
    std::cout << buffer.data() << std::endl;
  });

  return 0;
}
