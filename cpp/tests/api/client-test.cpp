#include <blazingdb/protocol/api.h>

int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
  blazingdb::protocol::Client client(connection);

  blazingdb::protocol::Buffer buffer(
      reinterpret_cast<const std::uint8_t*>("BlazingDB PROTOCOL"), 19);

  client.send(buffer);

  return 0;
}
