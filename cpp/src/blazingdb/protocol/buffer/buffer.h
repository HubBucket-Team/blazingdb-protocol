#ifndef BLAZINGDB_PROTOCOL_BUFFER_BUFFER_H_
#define BLAZINGDB_PROTOCOL_BUFFER_BUFFER_H_

#include <cstdint>

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

}  // namespace protocol
}  // namespace blazingdb

#endif
