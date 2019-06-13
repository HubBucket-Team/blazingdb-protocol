#ifndef BLAZINGDB_PROTOCOL_UTILITIES_IO_UTIL_H
#define BLAZINGDB_PROTOCOL_UTILITIES_IO_UTIL_H

#include <iostream>
#include <exception>

#include <blazingdb/protocol/buffer/buffer.h>

namespace blazingdb {
namespace protocol {
namespace util {

    void read_all(int descriptor, void * buffer, size_t size);
    void write_all(int descriptor, void * buffer, size_t size);

    void read_buffer(int descriptor, Buffer & buffer);
    void write_buffer(int descriptor, const Buffer & buffer);

}
}
}

#endif  // BLAZINGDB_PROTOCOL_UTILITIES_IO_UTIL_H