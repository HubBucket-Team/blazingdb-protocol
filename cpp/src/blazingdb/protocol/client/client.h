#ifndef BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_
#define BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_

#include <memory>
#include <blazingdb/protocol/buffer/buffer.h>
#include <blazingdb/protocol/connection/connection.h>
#include "flatbuffers/flatbuffers.h"

namespace blazingdb {
namespace protocol {

class Client {
public:
  explicit Client(const ConnectionAddress &connectionAddress);

  Buffer send(const Buffer &buffer);

  Buffer send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer);

private:
  int sock;
  std::string address;
  int port;
  struct sockaddr_in server;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
