#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"

namespace blazingdb {
namespace protocol {


namespace calcite {

class DMLResponseMessage : public StringTypeMessage<calcite::DMLResponse> {
public:
  DMLResponseMessage(const std::string &logicalPlan, const std::int64_t time)
    : time_{time}, StringTypeMessage<calcite::DMLResponse>(logicalPlan) {}

  DMLResponseMessage(const uint8_t *buffer)
    : StringTypeMessage<calcite::DMLResponse>("") {
    auto pointer       = flatbuffers::GetRoot<calcite::DMLResponse>(buffer);
    auto string_buffer = pointer->logicalPlan();
    time_              = pointer->time();
    string_value       = std::string{string_buffer->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{0};
    auto                           root_offset =
      calcite::CreateDMLResponseDirect(builder, string_value.c_str(), time_);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getLogicalPlan() { return string_value; }

  std::int64_t getTime() { return time_; }

private:
  std::int64_t time_;
};

class DMLRequestMessage : public IMessage {
public:

  DMLRequestMessage(const std::string& query) : IMessage(), query (query){
  }

  DMLRequestMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::calcite::DMLRequest>(buffer);
    query = std::string{pointer->query()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {

    flatbuffers::FlatBufferBuilder builder{0};
    //1. query.length();
    auto string_offset = builder.CreateString(query);
    auto root_offset = calcite::CreateDMLRequest(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getQuery() const {
    return query;
  }

private:
  std::string query;
};



class DDLRequestMessage : public IMessage {
public:

  DDLRequestMessage(const std::string& query) : IMessage(), query (query){
  }

  DDLRequestMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::calcite::DMLRequest>(buffer);
    query = std::string{pointer->query()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto string_offset = builder.CreateString(query);
    auto root_offset = calcite::CreateDMLRequest(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getQuery() const {
    return query;
  }

private:
  std::string query;
};


}
}
}
