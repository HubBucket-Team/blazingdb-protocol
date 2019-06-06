#ifndef BLAZINGDB_PROTOCOL_CONNECTION_TCP_CONNECTION_H_
#define BLAZINGDB_PROTOCOL_CONNECTION_TCP_CONNECTION_H_

#ifndef USE_UNIX_SOCKETS

#include "connection.h"

namespace blazingdb {
namespace protocol {

class TCPConnection : public Connection {
public:
  explicit TCPConnection(const ConnectionAddress &connectionAddress);

  ~TCPConnection() override;

  void initialize() const noexcept final;

  TCPConnection(const TCPConnection &)  = delete;
  TCPConnection(const TCPConnection &&) = delete;
  void operator=(const TCPConnection &) = delete;
  void operator=(const TCPConnection &&) = delete;

public:
  const std::string ip_;
};

}  // namespace protocol
}  // namespace blazingdb

#endif

#endif
