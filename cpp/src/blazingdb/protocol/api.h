#ifndef BLAZINGDB_PROTOCOL_API_H_
#define BLAZINGDB_PROTOCOL_API_H_

#include <cstring>
#include <stdexcept>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace blazingdb {
namespace protocol {

class Buffer {
public:
  Buffer(const std::uint8_t *const data, const std::size_t size)
      : data_(data), size_(size) {}

  const std::uint8_t *data() const { return data_; }

  std::size_t size() const { return size_; }

  Buffer slice(const std::ptrdiff_t offset) const {
    return {data_ + offset, size_ - static_cast<std::size_t>(offset)};
  }

private:
  const std::uint8_t *const data_;
  std::size_t size_;
};

namespace {

class StackBuffer : public Buffer {
public:
  static const std::size_t MAX_SIZE = 4096;

  StackBuffer()
      : Buffer(static_cast<const std::uint8_t *const>(actual_data_), MAX_SIZE),
        actual_data_{0} {}

  std::uint8_t *data() { return static_cast<std::uint8_t *>(actual_data_); }

private:
  std::uint8_t actual_data_[MAX_SIZE];
};

}  // namespace

class File {
public:
  virtual ~File() = default;

  virtual int fd() const = 0;

  File() = default;
  File(const File &) = delete;
  File(const File &&) = delete;
  void operator=(const File &) = delete;
  void operator=(const File &&) = delete;
};

class Connection : public File {
public:
  Connection(const int fd, const std::string &path)
      : fd_(fd), addr_{0, {}}, unused_{0} {
    bzero(&addr_, sizeof(addr_));
    addr_.sun_family = AF_UNIX;
    std::strncpy(static_cast<char *>(addr_.sun_path), path.c_str(),
                 path.size());
  }

  ~Connection() override = default;

  int fd() const final { return fd_; }

  __CONST_SOCKADDR_ARG address() const {
    return reinterpret_cast<__CONST_SOCKADDR_ARG>(&addr_);
  }

  socklen_t length() const { return sizeof(addr_); }

  const char (&unused() const)[6] { return unused_; }

  Connection(const Connection &) = delete;
  Connection(const Connection &&) = delete;
  void operator=(const Connection &) = delete;
  void operator=(const Connection &&) = delete;

protected:
  int fd_;

private:
  struct sockaddr_un addr_;
  char unused_[6];
};

class Server {
public:
  explicit Server(const Connection &connection) : connection_(connection) {
    unlink(static_cast<const char *>(connection_.address()->sa_data));

    if (bind(connection_.fd(), connection_.address(), connection_.length()) ==
        -1) {
      throw std::runtime_error("bind error");
    }

    if (listen(connection_.fd(), 5) == -1) {
      throw std::runtime_error("listen error");
    }
  }

  template <class Callable>
  void handle(Callable &&callback) const {
    for (;;) {
      int fd = accept4(connection_.fd(), nullptr, nullptr, SOCK_CLOEXEC);

      if (fd == -1) { throw std::runtime_error("accept error"); }

      StackBuffer buffer;
      ssize_t nread = read(fd, buffer.data(), buffer.size());

      if (nread > 0) {
        Buffer responseBuffer =
            callback(Buffer(buffer.data(), static_cast<std::size_t>(nread)));

        ssize_t written_bytes =
            write(fd, responseBuffer.data(), responseBuffer.size());

        if (static_cast<std::size_t>(written_bytes) != responseBuffer.size()) {
          throw std::runtime_error("write error");
        }
      } else if (nread == -1) {
        throw std::runtime_error("error read");
      } else if (nread == 0) {
        close(fd);
      } else {
        throw std::runtime_error("unreachable");
      }
    }
  }

private:
  const Connection &connection_;
};

// TODO(gcca): serializable through output stream
template <class Frame>
class Protocol {
public:
  //// TODO(gcca): reading input stream

  const Frame &frameFrom(const Buffer &buffer) const {
    const auto *frame = reinterpret_cast<const Frame *>(buffer.data());
    return *frame;
  }

  const Buffer payloadFrom(const Buffer &buffer) const {
    return buffer.slice(static_cast<std::ptrdiff_t>(sizeof(Frame)));
  }
};

template <class Frame>
class ProtocolRoutingServer : public Server {
  using Protocol_ = Protocol<Frame>;

public:
  explicit ProtocolRoutingServer(const Connection &connection,
                                 const Protocol_ &protocol)
      : Server(connection), protocol_(protocol) {}

  template <class Callable>
  void handle(Callable &&callback) const {
    Server::handle([this, &callback](const Buffer &buffer) -> void {
      const Frame &frame = protocol_.frameFrom(buffer);

      if (frame.kind < 0) { return; }

      const Buffer payloadBuffer = protocol_.payloadFrom(buffer);

      callback(payloadBuffer);
    });
  }

  template <class RequestFB, class Callable>
  void Register(Callable &callback) {
    callback(nullptr);
  }

private:
  const Protocol_ &protocol_;
};

class Client {
public:
  explicit Client(const Connection &connection) : connection_(connection) {
    int result =
        connect(connection.fd(), connection.address(), connection.length());

    if (result == -1) { throw std::runtime_error("connect error"); }
  }

  Buffer send(const Buffer &buffer) {
    ssize_t written_bytes =
        write(connection_.fd(), buffer.data(), buffer.size());

    if (static_cast<std::size_t>(written_bytes) != buffer.size()) {
      throw std::runtime_error("write error");
    }

    StackBuffer responseBuffer;
    ssize_t nread =
        read(connection_.fd(), responseBuffer.data(), responseBuffer.size());

    if (nread == -1) { throw std::runtime_error("error read"); }

    return responseBuffer;
  }

private:
  const Connection &connection_;
};

class UnixSocketConnection : public Connection {
public:
  explicit UnixSocketConnection(const std::string &path)
      : Connection(socket(AF_UNIX, SOCK_STREAM, 0), path) {
    if (fd_ == -1) { throw std::runtime_error("socket error"); }
  }

  ~UnixSocketConnection() override { close(fd_); }

  UnixSocketConnection(const UnixSocketConnection &) = delete;
  UnixSocketConnection(const UnixSocketConnection &&) = delete;
  void operator=(const UnixSocketConnection &) = delete;
  void operator=(const UnixSocketConnection &&) = delete;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
