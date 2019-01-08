#ifndef BLAZINGDB_PROTOCOL_CONNECTION_CONNECTION_H_
#define BLAZINGDB_PROTOCOL_CONNECTION_CONNECTION_H_

#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

namespace blazingdb {
namespace protocol {

class OS {
public:
  virtual ~OS() = default;

  // throws
  virtual void unlink(const std::string &path) = 0;
};

class File {
public:
  virtual ~File() = default;

  virtual int fd() const = 0;

  File()              = default;
  File(const File &)  = delete;
  File(const File &&) = delete;
  void operator=(const File &) = delete;
  void operator=(const File &&) = delete;
};

class Connection : public File {
public:
  Connection(const int fd, const std::string &path)
      : fd_(fd), addr_{0, {}}, unused_{0} {
    bzero(&addr_, sizeof(addr_));
    addr_.sin_family      = AF_INET;
    addr_.sin_addr.s_addr = INADDR_ANY;
    addr_.sin_port = htons(static_cast<std::uint16_t>(atoi(path.c_str())));
  }

  ~Connection() override = default;

  virtual void initialize() const noexcept = 0;

  int fd() const final { return fd_; }

  __CONST_SOCKADDR_ARG address() const {
    return reinterpret_cast<__CONST_SOCKADDR_ARG>(&addr_);
  }

  socklen_t length() const { return sizeof(addr_); }

  const char (&unused() const)[6] { return unused_; }

  Connection(const Connection &)  = delete;
  Connection(const Connection &&) = delete;
  void operator=(const Connection &) = delete;
  void operator=(const Connection &&) = delete;

protected:
  int                fd_;
  struct sockaddr_in addr_;

private:
  char unused_[6];
};

}  // namespace protocol
}  // namespace blazingdb

#endif
