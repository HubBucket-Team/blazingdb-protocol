# blazingdb-protocol
BlazingDB Protocol &amp; Messages


## Build / Install 

Here are the steps to do so, including the necessary dependencies, just be sure to have:

- a C++11 compiler (gcc 5+, clang 3.8+)
- CMake 3.3+

### Install dependencies

Install Flatbuffers

```
git clone https://github.com/google/flatbuffers.git
cd flatbuffers && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/flatbuffers_install ..
make -j8 install  /path/flatbuffers_install/
```

## Build Examples

We provide some examples on how to send/receive data using blazingdb-protocol.
You can build the examples turning the corresponding CMake flag BUILD_EXAMPLES.

`cmake  -DFLATBUFFERS_HOME=/path/flatbuffers_install/  -DBUILD_EXAMPLES=TRUE ..` 
