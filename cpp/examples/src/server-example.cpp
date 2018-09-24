#include <iostream>

#include <protocol/api.h>


#include "flatbuffers/flatbuffers.h"
#include "QueryMessage_generated.h"

namespace blazingdb {
namespace protocol {

class BlazingQueryMessage : public IMessage {
public: 
  BlazingQueryMessage() = default;

  ~BlazingQueryMessage() = default;

  /**
   * @brief Construct a String message using a raw memory coming from network.
   */

  BlazingQueryMessage(const Buffer& buffer) 
    : statement_{ GetQueryMessage(buffer.data())->statement()->c_str() },
      authorization_{ GetQueryMessage(buffer.data())->authorization()->c_str() }
  {
  }

  BlazingQueryMessage(std::string statement, std::string authorization)
    : IMessage{},
      statement_{std::move(statement)},
      authorization_{std::move(authorization)} {}

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    std::lock_guard<std::mutex> lock{mutex_};
    flatbuffers::FlatBufferBuilder builder{1024};

    auto statement_string_data = builder.CreateString(statement_);
    auto authorization_string_data = builder.CreateString(authorization_);
    QueryMessageBuilder tmp_builder{builder};
    tmp_builder.add_statement(statement_string_data);
    tmp_builder.add_authorization(authorization_string_data);

    FinishQueryMessageBuffer(builder, tmp_builder.Finish());

    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  std::string statement() const { return statement_; }

  std::string authorization() const { return authorization_; }


private:
  std::string statement_{""};
  std::string authorization_{""};

};

}
}

using namespace blazingdb::protocol;

int main() {
  UnixSocketConnection connection("/tmp/socket");
  Server server(connection);

  server.handle([](const Buffer &buffer) {
    BlazingQueryMessage message{buffer};

    std::cout << message.statement() << std::endl;
    std::cout << message.authorization() << std::endl;
  });

  return 0;
}
// python-client (DDL)
//  - bz.add_schema_db()
//  - bz.remove_schema_db()
//  - bz.register_table_schema()
//  - bz.unregister_table_schema()

// python-client (DML)
//  - bz.run_sql(sql, ipc_handlers) -> connection {resource_token}

// python:
// - get_result(resource_token) -> response_token


// python-client --------SQL(DML/DDL)------> cpp-orquestrator
// cpp-orquestrator -----SQL(DML/DDL)------> java-calcite
// java-calcite ---------RelationalAlgebra------> cpp-orquestrator

// 