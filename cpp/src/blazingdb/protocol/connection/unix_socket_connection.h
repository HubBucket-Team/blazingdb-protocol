#ifndef BLAZINGDB_PROTOCOL_CONNECTION_UNIX_SOCKET_CONNECTION_H_
#define BLAZINGDB_PROTOCOL_CONNECTION_UNIX_SOCKET_CONNECTION_H_

#include "connection.h"

namespace blazingdb {
namespace protocol {

class UnixSocketConnection : public Connection {
public:
  explicit UnixSocketConnection(const std::string &path);

  ~UnixSocketConnection() override;

  void initialize() const noexcept final;

  UnixSocketConnection(const UnixSocketConnection &) = delete;
  UnixSocketConnection(const UnixSocketConnection &&) = delete;
  void operator=(const UnixSocketConnection &) = delete;
  void operator=(const UnixSocketConnection &&) = delete;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
