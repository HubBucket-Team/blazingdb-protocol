#!/bin/sh
echo "run calcite service"
cd java/
java -cp target/blazingdb-protocol.jar com.blazingdb.protocol.examples.server.CalciteServiceExample 
