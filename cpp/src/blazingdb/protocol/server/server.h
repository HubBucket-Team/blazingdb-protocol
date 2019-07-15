#ifndef BLAZINGDB_PROTOCOL_SERVER_SERVER_H_
#define BLAZINGDB_PROTOCOL_SERVER_SERVER_H_

#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <utility>

#include <blazingdb/protocol/buffer/buffer.h>
#include <blazingdb/protocol/connection/connection.h>

namespace blazingdb {
namespace protocol {
 
class Server {
public:
  explicit Server(int tcp_port);

  template <class Callable>
  void handle(Callable &&callback) const {
    _Start(_MakeHandler(callback));
  }

private:
  int sockfd;
  struct sockaddr_in serverAddress;

  struct _HandlerBase;
  using __HandlerBaseType = std::shared_ptr<_HandlerBase>;

  void _Start[[noreturn]](const __HandlerBaseType &) const;

  struct _HandlerBase {
    inline virtual ~_HandlerBase() = default;
    virtual Buffer call(const Buffer &) = 0;
  };

  template <class Callable>
  class _Handler : public _HandlerBase {
  public:
    _Handler(Callable &&callback)
        : callback_(std::forward<Callable>(callback)) {}

    Buffer call(const Buffer &buffer) { return callback_(buffer); }

  private:
    Callable callback_;
  };

  template <class Callable>
  std::shared_ptr<_Handler<Callable>> _MakeHandler(Callable &&callback) const {
    return std::make_shared<_Handler<Callable>>(
        std::forward<Callable>(callback));
  }
};

}  // namespace protocol
}  // namespace blazingdb

#endif
