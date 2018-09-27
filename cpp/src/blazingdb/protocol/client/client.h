#ifndef BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_
#define BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_

#include "../buffer/buffer.h"
#include "../connection/connection.h"

namespace blazingdb {
namespace protocol {

class Client {
public:
  explicit Client(const Connection &connection);

  Buffer send(const Buffer &buffer);

private:
  const Connection &connection_;
};

}  // namespace protocol
}  // namespace blazingdb

#endif
