#ifndef BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_
#define BLAZINGDB_PROTOCOL_CLIENT_CLIENT_H_

#include <memory>
#include "../buffer/buffer.h"
#include "../connection/connection.h"
#include "flatbuffers/flatbuffers.h"

namespace blazingdb {
namespace protocol {

class Client {
public:
  explicit Client(const Connection &connection);

  Buffer send(const Buffer &buffer);

  Buffer send(std::shared_ptr<flatbuffers::DetachedBuffer> &buffer);

private:
  const Connection &connection_;
};

class ZeroMqClient {
public:
  explicit ZeroMqClient(const std::string &connection);
 
 ~ZeroMqClient(); 

  Buffer send(const Buffer &buffer);

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};


}  // namespace protocol
}  // namespace blazingdb

#endif
