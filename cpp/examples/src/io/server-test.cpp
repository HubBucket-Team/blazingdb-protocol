#include <iostream>

#include <blazingdb/protocol/api.h>

#include <blazingdb/protocol/message/io/file_system.h>


auto process(const blazingdb::protocol::Buffer &buffer)
                     -> blazingdb::protocol::Buffer {

     blazingdb::message::io::FileSystemRegisterRequestMessage message(buffer.data());
     std::cout << message.getAuthority() << std::endl;
     std::cout << message.getRoot() << std::endl;
     
     return blazingdb::protocol::Buffer(
         reinterpret_cast<const std::uint8_t *>("BlazingDB Response"), 18);
   }

int main() {
   blazingdb::protocol::ZeroMqServer server("ipc:///tmp/socket");


   server.handle(&process);

  return 0;
}
