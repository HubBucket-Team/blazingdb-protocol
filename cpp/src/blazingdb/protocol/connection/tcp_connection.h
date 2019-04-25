#ifndef BLAZINGDB_PROTOCOL_CONNECTION_TCP_CONNECTION_H_
#define BLAZINGDB_PROTOCOL_CONNECTION_TCP_CONNECTION_H_

#include "connection.h"

namespace blazingdb {
namespace protocol {

class TCPConnection : public NetConnection {
public:
  explicit TCPConnection(const std::string &ip, const std::string &port);

  ~TCPConnection() override;

  void initialize() const noexcept final;

  TCPConnection(const TCPConnection &)  = delete;
  TCPConnection(const TCPConnection &&) = delete;
  void operator=(const TCPConnection &) = delete;
  void operator=(const TCPConnection &&) = delete;

private:
  const std::string ip_;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
