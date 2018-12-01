#include "rex/base/RexData.h"

namespace blazingdb {
namespace protocol {
namespace dto {

void RexData::setValue(std::string& value) {
    this->value = value;
}

void RexData::setValue(std::string&& value) {
    this->value = std::move(value);
}

int8_t RexData::getSignedInteger08() {
    return (int8_t) stol(value);
}

uint8_t RexData::getUnsignedInteger08() {
    return (uint8_t) stoul(value);
}

int16_t RexData::getSignedInteger16() {
    return (int16_t) stol(value);
}

uint16_t RexData::getUnsignedInteger16() {
    return (uint16_t) stoul(value);
}

int32_t  RexData::getSignedInteger32() {
    return (int32_t) stol(value);
}

uint32_t RexData::getUnsignedInteger32() {
    return (uint32_t) stoul(value);
}

int64_t RexData::getSignedInteger64() {
    return (int64_t) stoll(value);
}

uint64_t RexData::getUnsignedInteger64() {
    return (uint64_t) stoull(value);
}

float RexData::getFloat() {
    return stof(value);
}

double RexData::getDouble() {
    return stod(value);
}

std::string RexData::getString() {
    return value;
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
