
# blazingdb-protocol
BlazingDB Protocol &amp; Messages

## Build / Install 

Here are the steps to do so, including the necessary dependencies, just be sure to have:

- a C++11 compiler (gcc 5.5+, clang 3.8+)
- CMake 3.11+
- Java 8+
- Python 3.5+

### Install dependencies

Install Flatbuffers

```
git clone https://github.com/google/flatbuffers.git
cd flatbuffers && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8 install
```

## Build and test blazingdb-protocol library

```
./build_libraries.sh

```
### Build cpp blazingdb-protocol library

```
cd cpp && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/some_directory/protocol_install \
      -DFLATBUFFERS_INSTALL_DIR=/home/aocsa/flatbuffers_install \
      .. 
make 
```

## Initialize Services

Initialize blazingsql mock services

```
./run_all_mock_services.sh
```

Initialize blazingsql mock services individually
```
./run_calcite_mock.sh
./run_ral_mock.sh
./run_orchestrator_mock.sh
```

## Use blazingsql python client 

```
cd python/examples/
python3 py-connector.py

```
