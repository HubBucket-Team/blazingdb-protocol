#include "io_util.h"

#include <iostream>
#include <exception>
#include <string>
#include <unistd.h>

namespace blazingdb {
namespace protocol {
namespace util {

    void read_all(int descriptor, void * buffer, size_t size){
        char * buffer_position = static_cast<char*>(buffer);
        size_t position = 0;
        size_t read_size = 1024 * 1024 * 1024;
        while(position < size){
            if(position + read_size > size){
                read_size = size - position;
            }
            int n = read(descriptor, buffer_position, read_size);
            position += n;
            buffer_position += n;
            if(n == 0){
                throw std::runtime_error{"ERROR: was not able to read the total amount of " + std::to_string(size)};                
            }
        }
    }

    void write_all(int descriptor, void * buffer, size_t size){
        char * buffer_position = static_cast<char*>(buffer);
        size_t position = 0;
        size_t read_size = 1024 * 1024 * 1024;
        while(position < size){
            if(position + read_size > size){
                read_size = size - position;
            }
            int n = write(descriptor, buffer_position, read_size);
            position += n;
            buffer_position += n;
            if(n == 0){
                throw std::runtime_error{"ERROR: was not able to write the total amount of " + std::to_string(size)};                
            }
        }
    }

    void read_buffer(int descriptor, Buffer & buffer){
        uint32_t response_buffer_length;
        
        std::cout << "qqqqqqqqqqqqq" << std::endl;
        
        
        read_all(descriptor, (void*)&response_buffer_length, sizeof(uint32_t));
        
        std::cout << "dentrlen es response_buffer_length: " << response_buffer_length << std::endl;
        
        buffer.resize(response_buffer_length);
        read_all(descriptor, (void*)buffer.data(), response_buffer_length);
        
        std::cout << "ultimo del read ese util  " <<std::endl;
    
    }
    void write_buffer(int descriptor, const Buffer & buffer){
        int buffer_length = buffer.size();
        write_all(descriptor, (void*)&buffer_length, sizeof(int));
        write_all(descriptor, (void*)buffer.data(), buffer_length);        
    }

}
}
}