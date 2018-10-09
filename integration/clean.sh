#!/bin/sh
CWD="$(pwd)"

echo "clean calcite service"
cd services/calcite-service && mvn clean 

cd $CWD

echo "clean cpp services"
cd services/cpp && rm -rf build 

echo "done blazingdb-protocol clean"
