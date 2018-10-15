#ifndef BLAZINGDB_PROTOCOL_BUFFER_BUFFER_H_
#define BLAZINGDB_PROTOCOL_BUFFER_BUFFER_H_

#include <cstdint>
#include <string>

namespace blazingdb {
namespace protocol {

class Buffer : public std::basic_string<std::uint8_t> {
public:
  Buffer()
      : std::basic_string<std::uint8_t >() {}

  Buffer(const std::uint8_t *const data, const uint32_t size)
      : std::basic_string<std::uint8_t >(data, size) {}

  Buffer slice(const std::ptrdiff_t offset) const {
    return {this->data() + offset, this->size() - static_cast<uint32_t>(offset)};
  }
};

}  // namespace protocol
}  // namespace blazingdb

#endif
