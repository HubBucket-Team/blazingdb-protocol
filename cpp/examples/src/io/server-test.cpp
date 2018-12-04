#include <iostream>

#include <blazingdb/protocol/api.h>

#include <blazingdb/protocol/message/io/file_system.h>

int main() {
   blazingdb::protocol::UnixSocketConnection connection(
       {"/tmp/socket", std::allocator<char>()});
   blazingdb::protocol::Server server(connection);

   server.handle([](const blazingdb::protocol::Buffer &buffer)
                     -> blazingdb::protocol::Buffer {

     blazingdb::message::io::FileSystemRegisterRequestMessage message(buffer.data());
     std::cout << message.getAuthority() << std::endl;
     std::cout << message.getRoot() << std::endl;
     
     return blazingdb::protocol::Buffer(
         reinterpret_cast<const std::uint8_t *>("BlazingDB Response"), 18);
   });

  return 0;
}
