#include <blazingdb/protocol/message/io/file_system.h>

#include <gtest/gtest.h>

using blazingdb::message::io::CommunicationContextSchema;
using blazingdb::message::io::CommunicationNodeSchema;
using blazingdb::message::io::CsvFileSchema;
using blazingdb::message::io::FileSystemBlazingTableSchema;
using blazingdb::message::io::FileSystemDMLRequestMessage;
using blazingdb::message::io::FileSystemTableGroupSchema;
using blazingdb::message::io::ParquetFileSchema;
using blazingdb::protocol::FileSchemaType;

inline static std::vector<CommunicationNodeSchema> MakeCommunicationNodes() {
  std::vector<std::int8_t> buffer{1, 2, 3};
  return {{buffer}};
}

inline static CommunicationContextSchema MakeCommunicationContext() {
  std::vector<CommunicationNodeSchema> nodes{MakeCommunicationNodes()};
  return {nodes, 0, 12345};
}

inline static std::vector<FileSystemBlazingTableSchema> MakeTables() {
  const CsvFileSchema csvFilesSchema{
      "csv file 1", ",", "\0", 1, {"name 1", "name 2"}, {1, 2}};

  const ParquetFileSchema parquetFileSchema{
      "parquet file 1", {1, 2, 3}, {4, 5, 6}};

  const FileSystemBlazingTableSchema filesystemBlazingTableSchema{
        "table 1",
       FileSchemaType::FileSchemaType_CSV,
       csvFilesSchema,
       parquetFileSchema};

  return {filesystemBlazingTableSchema};
}
	std::vector<std::string> names;
	std::vector<uint64_t> calciteToFileIndices;
	std::vector<int> types;
	std::vector<uint64_t> numRowGroups;
  std::vector<std::string> files;
  std::string csvDelimiter;
  std::string csvLineTerminator;
  uint32_t csvSkipRows;
inline static FileSystemTableGroupSchema MakeSchema() {
  return {MakeTables(), "testTableGroup"};
}

TEST(DMLRequestMessage, CheckSerialization) {
  const std::string                statement      = "select statement;";
  const FileSystemTableGroupSchema schema         = MakeSchema();
  const CommunicationContextSchema communicationContext = MakeCommunicationContext();

  FileSystemDMLRequestMessage message{statement, schema, communicationContext, 0};

  std::shared_ptr<flatbuffers::DetachedBuffer> detachedBuffer =
      message.getBufferData();

  // CHECK

  using blazingdb::protocol::io::FileSystemDMLRequest;

  const FileSystemDMLRequest * request =
      flatbuffers::GetRoot<FileSystemDMLRequest>(detachedBuffer->data());

  // Check statement

  const std::string expectedStatement = "select statement;";
  EXPECT_EQ(expectedStatement, request->statement()->str());

  // Check tableGroup

  using blazingdb::protocol::io::FileSystemTableGroup;
  const FileSystemTableGroup * tableGroup = request->tableGroup();

  const std::string expectedName = "testTableGroup";
  EXPECT_EQ(expectedName, tableGroup->name()->str());

  using blazingdb::protocol::io::FileSystemBlazingTable;
  const FileSystemBlazingTable * table = tableGroup->tables()->Get(0);

  const std::string expectedTableName = "table 1";
  EXPECT_EQ(expectedTableName, table->name()->str());

  // Check communicationContext
}

TEST(DMLRequestMessage, CheckConversion) {
  const std::string                statement      = "select statement;";
  const FileSystemTableGroupSchema schema         = MakeSchema();
  const CommunicationContextSchema communicationContext = MakeCommunicationContext();

  FileSystemDMLRequestMessage message{statement, schema, communicationContext, 0};

  std::shared_ptr<flatbuffers::DetachedBuffer> detachedBuffer =
      message.getBufferData();

  FileSystemDMLRequestMessage resultMessage(detachedBuffer->data());

  EXPECT_EQ("select statement;", resultMessage.statement());

  FileSystemTableGroupSchema resultTableGroup = resultMessage.tableGroup();

  EXPECT_EQ("testTableGroup", resultTableGroup.name);

  CommunicationContextSchema resultCommunicationContext =
      resultMessage.communicationContext();

  EXPECT_EQ(0, resultCommunicationContext.masterIndex);
  EXPECT_EQ(12345, resultCommunicationContext.token);

  std::vector<std::int8_t> expectedBuffer{1, 2, 3};
  EXPECT_EQ(expectedBuffer, resultCommunicationContext.nodes[0].buffer);
}
