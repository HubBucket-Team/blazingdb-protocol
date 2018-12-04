#pragma once

#include <string>
#include <cstdint>

namespace blazingdb {
namespace protocol {
namespace dto {

class RexData {
public:
    void setValue(std::string& value);

    void setValue(std::string&& value);

public:
    int8_t  getSignedInteger08();

    uint8_t getUnsignedInteger08();

    int16_t getSignedInteger16();

    uint16_t getUnsignedInteger16();

    int32_t  getSignedInteger32();

    uint32_t getUnsignedInteger32();

    int64_t  getSignedInteger64();

    uint64_t getUnsignedInteger64();

    float  getFloat();

    double getDouble();

    std::string getString();

private:
    std::string value {""};
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
