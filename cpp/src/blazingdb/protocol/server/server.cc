#include "server.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <thread>

#include <blazingdb/protocol/utilities/io_util.h>

namespace blazingdb {
namespace protocol {

Server::Server(int tcp_port) {
    
    int opt = 1;
    sockfd = socket(AF_INET,SOCK_STREAM, 0);
    
    if (sockfd == -1) {
        throw std::runtime_error("ERROR: Could not create the server socket");
    }
    
    int on = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) < 0) {
      throw std::runtime_error("set server socket option error");
    }
    
    memset(&serverAddress,0,sizeof(serverAddress));
    
    serverAddress.sin_family      = AF_INET;
    serverAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    serverAddress.sin_port        = htons(tcp_port);
    
    if((bind(sockfd,(struct sockaddr *)&serverAddress, sizeof(serverAddress))) < 0){
        throw std::runtime_error("Server: bind error");
    }
    
    if(listen(sockfd, 100) < 0){
		throw std::runtime_error("Server: listen error");
	}
}

void Server::_Start(const __HandlerBaseType &handler) const {
  for (;;) {
    struct sockaddr_in client_address;
    socklen_t client_address_size = sizeof(client_address);
    
    int fd = accept4(sockfd, (struct sockaddr*)&client_address, &client_address_size, SOCK_CLOEXEC);
    
    if (fd == -1) {
        throw std::runtime_error("accept error in server");
    }
    
    const std::string client_ip = inet_ntoa(client_address.sin_addr);
    
    std::cout << "Server is handling the response for client: " << client_ip << std::endl;
    
    Buffer temp_buffer;
    util::read_buffer(fd, temp_buffer);

    Buffer response_buffer = handler->call(temp_buffer);

    util::write_buffer(fd, response_buffer);

    close(fd);    
  }
}

}  // namespace protocol
}  // namespace blazingdb
