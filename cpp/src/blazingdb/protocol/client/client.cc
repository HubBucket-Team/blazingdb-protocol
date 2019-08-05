#include "client.h"

#include <string>
#include <stdexcept>

#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <blazingdb/protocol/utilities/io_util.h>

namespace blazingdb {
namespace protocol {

Client::Client(const ConnectionAddress &connectionAddress) {
    address = connectionAddress.tcp_host;
    port = connectionAddress.tcp_port;
    
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        std::cout << "Could not create the client socket" << std::endl;
    }
    
    bzero(&server, sizeof(server));
    
    if((signed)inet_addr(address.c_str()) == -1) {
        struct hostent *he;
        struct in_addr **addr_list;
        if ((he = gethostbyname( address.c_str())) == NULL) {
            herror("gethostbyname");
            std::cout<<"Failed to resolve hostname\n";
            //return false;
        }
        
        addr_list = (struct in_addr **) he->h_addr_list;
        for(int i = 0; addr_list[i] != NULL; i++) {
            server.sin_addr = *addr_list[i];
            break;
        }
    }
    else {
        server.sin_addr.s_addr = inet_addr(address.c_str());
    }
    
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        close(sock);
        throw std::runtime_error("Connection to server failed.");
    }
    
    //TODO dtor close the socket
}

Client::~Client(){
    close(sock);
}


Buffer Client::send(const Buffer &buffer) {
  if(sock == -1) {
      std::cout << "ERROR: Invalid socket for buffer size:" << buffer.size() <<std::endl;
  }

  util::write_buffer(sock, buffer);
  
  Buffer response_buffer;
  util::read_buffer(sock, response_buffer);
  
  return response_buffer;
}

Buffer Client::send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
  return this->send(Buffer{buffer->data(), buffer->size()});
}


}  // namespace protocol
}  // namespace blazingdb
