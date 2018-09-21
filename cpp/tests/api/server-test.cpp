#include <protocol/api.h>

int main() {
  blazingdb::protocol::UnixSocketConnection conn("/tmp/socket");

  blazingdb::protocol::Server server(conn);

  server.prepare();

  server.handle([](const char *buffer, const std::size_t size) {
    std::cout << buffer << std::endl;
  });

  return 0;
}
