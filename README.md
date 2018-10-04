
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

## Build

### Build cpp-library
```
cd cpp
mkdir build && cd build
cmake ..
make -j8 
```
### Build java-library
```
cd java
mvn clean install
```

### Build python-library
```
cd python
python3 setup.py install --user
```

## Build Integration Tests

```
cd integration
bash build.sh
```

## Initialize Services

Initialize blazingsql services
```
cd integration
bash run.sh
```

Use blazingsql python client 

```
cd integration/clients/python-connector
python3 py-connector.py
```

