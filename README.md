# blazingdb-protocol
BlazingDB Protocol &amp; Messages

## Build / Install 

Here are the steps to do so, including the necessary dependencies, just be sure to have:

- a C++11 compiler (gcc 5.5+, clang 3.8+)
- CMake 3.11+
- Java 8+
- Python 3.5+

## Dependencies
- General dependencies: https://github.com/BlazingDB/blazingdb-toolchain

## Build and test blazingdb-protocol library

```
./build_libraries.sh

```
### Build cpp blazingdb-protocol library

```bash
cd blazingdb-protocol
mkdir build
CUDACXX=/usr/local/cuda-9.2/bin/nvcc cmake -DCMAKE_BUILD_TYPE=Debug \
      -DBUILD_TESTING=ON \
      -DBLAZINGDB_DEPENDENCIES_INSTALL_DIR=/foo/blazingsql/dependencies/ \
      -DCMAKE_INSTALL_PREFIX:PATH=/foo/blazingdb_protocol_install_dir/ \
      ..
make -j8
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
