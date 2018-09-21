#ifndef _BLAZINGDB_PROTOCOL_API_H
#define _BLAZINGDB_PROTOCOL_API_H

#include <iostream>
#include <stdexcept>
#include <string>

#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace blazingdb {
namespace protocol {

class Connection {
public:
  virtual int fd() const = 0;
  virtual const std::string &path() const = 0;
};

class Server {
public:
  Server(const Connection &connection) : connection(connection) {}

  void prepare() const {
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, connection.path().c_str(), connection.path().size());

    unlink(connection.path().c_str());

    if (bind(connection.fd(), (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      throw std::runtime_error("bind error");
    }

    if (listen(connection.fd(), 5) == -1) {
      throw std::runtime_error("listen error");
    }
  }

  template <class Callable>
  void handle(Callable &&callback) const {
    int fd = accept(connection.fd(), nullptr, nullptr);

    if (fd == -1) { throw std::runtime_error("accept error"); }

    char buffer[100];
    bzero(buffer, 100);

    int nread = read(fd, buffer, sizeof(buffer));

    if (nread > 0) {
      callback(buffer, nread);
    } else if (nread == -1) {
      throw std::runtime_error("error read");
    } else if (nread == 0) {
      std::cout << "EOF" << std::endl;
      close(fd);
    } else {
      throw std::runtime_error("unreachable");
    }
  }

private:
  const Connection &connection;
};

class Client {
public:
  Client(const Connection &connection) : connection(connection) {}

  void connect() const {
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, connection.path().c_str(), connection.path().size());

    int result =
        ::connect(connection.fd(), (struct sockaddr *)&addr, sizeof(addr));

    if (result == -1) { throw std::runtime_error("connect error"); }
  }

  void send(const std::string &message) {
    write(connection.fd(), message.c_str(), message.size());
  }

private:
  const Connection &connection;
};

class UnixSocketConnection : public Connection {
public:
  UnixSocketConnection(const std::string &path) : path_(path), fd_(0) {
    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ == -1) { throw std::runtime_error("socket error"); }
  }

  int fd() const final { return fd_; }

  const std::string &path() const final { return path_; }

private:
  const std::string path_;
  int fd_;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
