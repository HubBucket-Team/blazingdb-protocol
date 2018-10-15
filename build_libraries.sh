CWD="$(pwd)"

echo "Build cpp library"
cd cpp && mkdir -p build && cd build
cmake .. 
make clean
make -j8

cp src/blazingdb/protocol/all_generated.h ../src/blazingdb/protocol/generated/
echo "	copy updated compiled flatbuffer"
make clean
make -j8

cd $CWD
echo "Build java library"
cd java
mvn clean install

cd $CWD
# echo "Build python library"
# cd python/blazingdb/messages/
# flatc -p --gen-all --gen-object-api ../../../messages/all.fbs

cd $CWD
cd python/
python3 setup.py install --user
