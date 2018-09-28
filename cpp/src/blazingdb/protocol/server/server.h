#ifndef BLAZINGDB_PROTOCOL_SERVER_SERVER_H_
#define BLAZINGDB_PROTOCOL_SERVER_SERVER_H_

#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../buffer/buffer.h"
#include "../connection/connection.h"

namespace blazingdb {
namespace protocol {


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

class Server {
public:
  explicit Server(const Connection &connection);

  template <class Callable>
  void handle(Callable &&callback) const {
    _Start(_MakeHandler(callback));
  }

private:
  const Connection &connection_;

  struct _HandlerBase;
  using __HandlerBaseType = std::shared_ptr<_HandlerBase>;

  void _Start[[noreturn]](const __HandlerBaseType &) const;

  ssize_t _GetRequest(int fd, StackBuffer &buffer) const;

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
