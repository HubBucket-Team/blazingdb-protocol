#!/bin/sh
bash clean.sh

CWD="$(pwd)"

echo "Build calcite service"
cd services/calcite-service && mvn clean install

cd $CWD

echo "Build cpp services"
cd services/cpp && mkdir -p build && cd build

echo "cmake cpp services"
cmake ..

echo "make cpp services"
make -j8

echo "done blazingdb-protocol integration tests"
