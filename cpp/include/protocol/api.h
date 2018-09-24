#ifndef _BLAZINGDB_PROTOCOL_API_H
#define _BLAZINGDB_PROTOCOL_API_H

#include <cstring>
#include <stdexcept>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <flatbuffers/flatbuffers.h>
#include <mutex>

namespace blazingdb {
namespace protocol {

namespace {

class Buffer {
public:
  Buffer(const std::uint8_t *const data, const std::size_t size)
      : data_(data), size_(size) {}

  const std::uint8_t *data() const { return data_; }

  std::size_t size() const { return size_; }

private:
  const std::uint8_t *const data_;
  std::size_t size_;
};

class ManagedBuffer : public Buffer {
public:
  static const std::size_t MAX_SIZE = 4096;

  ManagedBuffer()
      : Buffer(static_cast<const std::uint8_t *const>(actual_data_), MAX_SIZE),
        actual_data_{0} {}

  std::uint8_t *data() { return static_cast<std::uint8_t *>(actual_data_); }

private:
  std::uint8_t actual_data_[MAX_SIZE];
};

class File {
public:
  virtual int fd() const = 0;
};

}  // namespace


class Connection : public File {
public:
  Connection(const int fd, const std::string &path) : fd_(fd), addr_{0, {}} {
    bzero(&addr_, sizeof(addr_));
    addr_.sun_family = AF_UNIX;
    std::strncpy(static_cast<char *>(addr_.sun_path), path.c_str(),
                 path.size());
  }

  int fd() const final { return fd_; }

  __CONST_SOCKADDR_ARG address() const {
    return reinterpret_cast<__CONST_SOCKADDR_ARG>(&addr_);
  }

  socklen_t length() const { return sizeof(addr_); }

protected:
  int fd_;

private:
  struct sockaddr_un addr_;
};

class Server {
public:
  explicit Server(const Connection &connection) : connection(connection) {
    unlink(static_cast<const char *>(connection.address()->sa_data));

    if (bind(connection.fd(), connection.address(), connection.length()) ==
        -1) {
      throw std::runtime_error("bind error");
    }

    if (listen(connection.fd(), 5) == -1) {
      throw std::runtime_error("listen error");
    }
  }

  // template <class Callable>
  // void handle(Callable &&callback) const {
  //   int fd = accept4(connection.fd(), nullptr, nullptr, SOCK_CLOEXEC);

  //   if (fd == -1) { throw std::runtime_error("accept error"); }

  //   ManagedBuffer buffer;

  //   int nread = read(fd, buffer.data(), buffer.size());

  //   if (nread > 0) {
  //     callback(Buffer(buffer.data(), nread));
  //   } else if (nread == -1) {
  //     throw std::runtime_error("error read");
  //   } else if (nread == 0) {
  //     close(fd);
  //   } else {
  //     throw std::runtime_error("unreachable");
  //   }
  // }

  template <class Callable>
  void handle(Callable &&callback) const {
    int fd = accept4(connection.fd(), nullptr, nullptr, SOCK_CLOEXEC);

    if (fd == -1) { throw std::runtime_error("accept error"); }

    ManagedBuffer buffer;

    int nread = read(fd, buffer.data(), buffer.size());

    if (nread > 0) {
      
      callback( std::move(Buffer(buffer.data(), nread)) );

    } else if (nread == -1) {
      throw std::runtime_error("error read");
    } else if (nread == 0) {
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
  explicit Client(const Connection &connection) : connection(connection) {
    int result =
        connect(connection.fd(), connection.address(), connection.length());

    if (result == -1) { throw std::runtime_error("connect error"); }
  }

  void send(const Buffer &buffer) {
    std::size_t written_bytes =
        write(connection.fd(), buffer.data(), buffer.size());

    if (written_bytes != buffer.size()) {
      throw std::runtime_error("write error");
    }
  }

  void send(const std::shared_ptr<flatbuffers::DetachedBuffer> &buffer) {
    std::size_t written_bytes =
        write(connection.fd(), buffer->data(), buffer->size());

    if (written_bytes != buffer->size()) {
      throw std::runtime_error("write error");
    }
  }
private:
  const Connection &connection;
};

class UnixSocketConnection : public Connection {
public:
  explicit UnixSocketConnection(const std::string &path)
      : Connection(socket(AF_UNIX, SOCK_STREAM, 0), path) {
    if (fd_ == -1) { throw std::runtime_error("socket error"); }
  }

  ~UnixSocketConnection() { close(fd_); }

  UnixSocketConnection(const UnixSocketConnection &) = delete;
  UnixSocketConnection(const UnixSocketConnection &&) = delete;
  void operator=(const UnixSocketConnection &) = delete;
  void operator=(const UnixSocketConnection &&) = delete;
};

class IMessage {
public:
  IMessage() = default;

  virtual ~IMessage() = default;

  virtual std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const = 0;

protected:
  mutable std::mutex mutex_{};
};

}  // namespace protocol
}  // namespace blazingdb

#endif
