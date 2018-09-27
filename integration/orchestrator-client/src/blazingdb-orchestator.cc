#include <iostream>

#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
 
namespace blazingdb {
namespace protocol { 
  
class CalciteClient {
public:
  CalciteClient(blazingdb::protocol::Connection & connection) : client {connection}
  {}

  std::string getLogicalPlan(std::string query) {
    DMLRequestMessage requestPayload{query};
    RequestMessage requestObject{calcite::MessageType_DML, requestPayload}; 
    
    auto bufferedData = requestObject.getBufferData();

    Buffer buffer{bufferedData->data(), 
                  bufferedData->size()};

    Buffer responseBuffer = client.send(buffer);
    ResponseMessage response{responseBuffer.data()};
    DMLResponseMessage responsePayload(response.getPayloadBuffer());
    return responsePayload.getLogicalPlan();
  }

  void updateSchema(std::string query) {
      DMLRequestMessage requestPayload{query};
      RequestMessage requestObject{calcite::MessageType_DDL, requestPayload}; 
      
      auto bufferedData = requestObject.getBufferData();

      Buffer buffer{bufferedData->data(), 
                    bufferedData->size()};

      Buffer responseBuffer = client.send(buffer);
      // ResponseMessage response{responseBuffer.data()};
      // DDLResponseMessage responsePayload(response.getPayloadBuffer());
      
  }

private:
  blazingdb::protocol::Client client;
};


}
}
using namespace blazingdb::protocol;


int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/socket");
  CalciteClient client{connection};
  
  {
     std::string query = "select * from orders";
    try {
        std::string logicalPlan = client.getLogicalPlan(query);
        std::cout << logicalPlan << std::endl;
    } catch (std::exception &error) {
        std::cout << error.what() << std::endl;
    }
  }
  // {
  //   std::string query = "cas * from orders";
  //   try {
  //       client.updateSchema(query);
        
  //   } catch (std::exception &error) {
  //       std::cout << error.what() << std::endl;
  //   }
  // }
  return 0;
}
