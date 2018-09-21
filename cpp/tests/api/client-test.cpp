#include <protocol/api.h>

int main() {
  blazingdb::protocol::UnixSocketConnection conn("/tmp/socket");

  blazingdb::protocol::Client client(conn);

  client.send("BlazingDB-3.0");

  return 0;
}
