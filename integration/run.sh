#!/bin/sh
CWD="$(pwd)"

echo "run calcite service"
cd services/calcite-service
java -jar target/calcite-service-1.0-SNAPSHOT.jar &
P1=$!

cd $CWD

cd services/cpp/build/ 

echo "running ral services"
./blazingdb_ral_service &
P2=$!

echo "running orchestator services"
./blazingdb_orchestator_service &
P3=$!

echo "running blazingdb services"

wait $P1 $P2 $P3
