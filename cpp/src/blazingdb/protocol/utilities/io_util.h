#pragma once

#include <iostream>
#include <exception>

#include "../buffer/buffer.h"

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