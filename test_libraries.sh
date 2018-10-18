#!/bin/sh
CWD="$(pwd)"

cd $CWD
echo "run calcite service"
cd java/
java -cp target/blazingdb-protocol.jar com.blazingdb.protocol.examples.server.CalciteServiceExample &
P1=$!
sleep 1 

cd $CWD
echo "running ral services"
cd cpp/build/ 
./bin/blazingdb_ral_service &
P2=$!
sleep 1 

cd $CWD
cd cpp/build/ 
echo "running orchestator services"
./bin/blazingdb_orchestator_service &
P3=$!
sleep 1 


echo "running blazingdb services"
wait $P1 $P2 $P3
