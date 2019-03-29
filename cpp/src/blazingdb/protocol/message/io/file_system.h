#pragma once

#include "../messages.h"

namespace blazingdb {
namespace message {
namespace io {

struct POSIX {};

enum DriverType {
  DriverType_UNDEFINED = 0,
  DriverType_LIBHDFS = 1,
  DriverType_LIBHDFS3 = 2,
};

struct HDFS {
  std::string host;
  int port;
  std::string user;
  DriverType driverType;
  std::string kerberosTicket;
};

enum EncryptionType {
  EncryptionType_UNDEFINED = 0,
  EncryptionType_NONE = 1,
  EncryptionType_AES_256 = 2,
  EncryptionType_AWS_KMS = 3,
};

struct S3 {
  std::string bucketName;
  EncryptionType encryptionType;
  std::string kmsKeyAmazonResourceName;
  std::string accessKeyId;
  std::string secretKey;
  std::string sessionToken;
};

//#warning, verificar no duplicados en all_generated.h
enum FileSystemType {
  FileSystemType_NONE = 0,
  FileSystemType_POSIX = 1,
  FileSystemType_HDFS = 2,
  FileSystemType_S3 = 3,
};

using namespace blazingdb::protocol;


class FileSystemDeregisterRequestMessage : public StringTypeMessage<::blazingdb::protocol::io::FileSystemDeregisterRequest> {
public:
  FileSystemDeregisterRequestMessage(const std::string& string_value)
      : StringTypeMessage<::blazingdb::protocol::io::FileSystemDeregisterRequest>(string_value)
  {
  }

  FileSystemDeregisterRequestMessage (const uint8_t* buffer)
      :  StringTypeMessage<::blazingdb::protocol::io::FileSystemDeregisterRequest>(buffer, &::blazingdb::protocol::io::FileSystemDeregisterRequest::authority)
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    return this->getBufferDataUsing(::blazingdb::protocol::io::CreateFileSystemDeregisterRequestDirect);
  }

  std::string getAuthority () {
    return string_value;
  }
};


class FileSystemRegisterRequestMessage : public IMessage {
public:
  FileSystemRegisterRequestMessage(const std::string &authority, const std::string &root)
      : IMessage(), authority{authority}, root{root}, type{FileSystemType_POSIX} {}

  FileSystemRegisterRequestMessage(const std::string &authority, const std::string &root, const HDFS &hdfs)
      : IMessage(), authority{authority}, root{root}, type{FileSystemType_HDFS}, hdfs{hdfs} {}

  FileSystemRegisterRequestMessage(const std::string &authority, const std::string &root, const S3 &s3)
      : IMessage(), authority{authority}, root{root}, type{FileSystemType_S3}, s3{s3} {}

  FileSystemRegisterRequestMessage(const uint8_t *buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<
        blazingdb::protocol::io::FileSystemRegisterRequest>(buffer);
    authority = std::string{pointer->authority()->c_str()};

    root = std::string{pointer->root()->c_str()};
    type = (FileSystemType)pointer->fileSystemConnection_type(); // FileSystemType: same as enum blazingdb::protocol::io::FileSystemConnection

    if (type ==  FileSystemType_HDFS) {
      auto hdfs_local = pointer->fileSystemConnection_as_HDFS();
      this->hdfs = HDFS{
        .host = hdfs_local->host()->c_str(),
        .port = hdfs_local->port(),
        .user = hdfs_local->user()->c_str(),
        .driverType = (DriverType)hdfs_local->driverType(),
        .kerberosTicket = hdfs_local->kerberosTicket()->c_str(),
      };
    }
    else if (type == FileSystemType_S3) {
      auto s3_local = pointer->fileSystemConnection_as_S3();
      this->s3 = S3{
        .bucketName = s3_local->bucketName()->c_str(),
        .encryptionType = (EncryptionType)s3_local->encryptionType(),
        .kmsKeyAmazonResourceName = s3_local->kmsKeyAmazonResourceName()->c_str(),
        .accessKeyId = s3_local->accessKeyId()->c_str(),
        .secretKey = s3_local->secretKey()->c_str(),
        .sessionToken = s3_local->sessionToken()->c_str(),
      };
    }
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder;
    if (type == FileSystemType_HDFS) {
      auto union_obj = blazingdb::protocol::io::CreateHDFSDirect(
          builder, hdfs.host.c_str(), hdfs.port, hdfs.user.c_str(),
          (blazingdb::protocol::io::DriverType)hdfs.driverType,
          hdfs.kerberosTicket.c_str());
      builder.Finish(CreateFileSystemRegisterRequestDirect(
          builder, authority.c_str(), root.c_str(),
          blazingdb::protocol::io::FileSystemConnection_HDFS, union_obj.Union()));
    } else if (type == FileSystemType_S3) {
      auto union_obj = blazingdb::protocol::io::CreateS3Direct(
          builder, s3.bucketName.c_str(),
          (blazingdb::protocol::io::EncryptionType)s3.encryptionType,
          s3.kmsKeyAmazonResourceName.c_str(), s3.accessKeyId.c_str(),
          s3.secretKey.c_str(), s3.sessionToken.c_str());

      builder.Finish(CreateFileSystemRegisterRequestDirect(
          builder, authority.c_str(), root.c_str(),
          blazingdb::protocol::io::FileSystemConnection_S3, union_obj.Union()));
    } else {
      builder.Finish(CreateFileSystemRegisterRequestDirect(
          builder,  authority.c_str(), root.c_str(),
          blazingdb::protocol::io::FileSystemConnection_POSIX, 0));
    }
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getAuthority() const {
    return authority;
  }
  std::string getRoot () const {
    return root;
  }

  HDFS getHdfs() const {
    return hdfs;
  }

  S3 getS3() const {
    return s3;
  }

  bool isLocal() const {
    return type == FileSystemType_POSIX;
  }

  bool isHdfs() const {
    return type == FileSystemType_HDFS;
  }

  bool isS3() const {
    return type == FileSystemType_S3;
  }


private:
  std::string authority;
  std::string root;
  FileSystemType type;

  // POSIX posix;
  HDFS hdfs;
  S3 s3;
};


std::vector<flatbuffers::Offset<flatbuffers::String>>  BuildeFlatStringList(flatbuffers::FlatBufferBuilder &builder, const std::vector<std::string> &strings)   {
  std::vector<flatbuffers::Offset<flatbuffers::String>> offsets;
  for (auto & str: strings) {
    offsets.push_back( builder.CreateString(str.data()));
  }
  return offsets;
}

struct CsvFileSchema {
  std::string path;
  std::string delimiter = "|";
  std::string line_terminator = "\n";
  int skip_rows = 0;
  std::vector<std::string> names;
  std::vector<int> dtypes;


  static flatbuffers::Offset<blazingdb::protocol::io::CsvFile> Serialize(flatbuffers::FlatBufferBuilder &builder, CsvFileSchema data) {
    std::vector<int> dtypes;
    std::vector<flatbuffers::Offset<flatbuffers::String>>  names = BuildeFlatStringList(builder, data.names);
    // std::vector<int32_t> &dtypes = data.dtypes; ??@todo

    return blazingdb::protocol::io::CreateCsvFile(builder, builder.CreateString(data.path.c_str()), builder.CreateString(data.delimiter.c_str()), builder.CreateString(data.line_terminator.c_str()), data.skip_rows, builder.CreateVector(names), builder.CreateVector( data.dtypes.data(),  data.dtypes.size()) );
  }

  static void Deserialize(const blazingdb::protocol::io::CsvFile *pointer, CsvFileSchema* schema) {
    schema->path =  std::string{pointer->path()->c_str()};
    if (std::string{pointer->delimiter()->c_str()}.length() > 0)
      schema->delimiter =  std::string{pointer->delimiter()->c_str()};
    if (std::string{pointer->lineTerminator()->c_str()}.length() > 0)
      schema->line_terminator =  std::string{pointer->lineTerminator()->c_str()};
    schema->skip_rows =  pointer->skipRows();

    auto ColumnNamesFrom = [](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *rawNames) -> std::vector<std::string> {
      std::vector<std::string> columnNames;
      for (const auto& rawName : *rawNames){
        auto name = std::string{rawName->c_str()};
        columnNames.push_back(name);
      }
      return columnNames;
    };
    auto ColumnTypesFrom = [](const flatbuffers::Vector<int32_t> *rawValues) -> std::vector<int> {
      std::vector<int> values;
      for (const auto& val : *rawValues){
        values.push_back(val);
      }
      return values;
    };
    schema->names = ColumnNamesFrom(pointer->names());
    schema->dtypes = ColumnTypesFrom(pointer->dtypes());
  }
};

class LoadCsvFileRequestMessage : public IMessage, CsvFileSchema {
public:
  LoadCsvFileRequestMessage(const std::string path,
      const std::string & delimiter,
			const std::string & line_terminator,
			int skip_rows,
			const std::vector<std::string> & names,
			const std::vector<int> & dtypes)
      : IMessage()
  {
     this->path = path;
     this->delimiter = delimiter;
     this->line_terminator = line_terminator;
     this->skip_rows = skip_rows;
     this->names = names;
     this->dtypes = dtypes;
  }

  LoadCsvFileRequestMessage(const uint8_t *buffer) : IMessage() {
    auto data = flatbuffers::GetRoot<blazingdb::protocol::io::CsvFile>(buffer);
    CsvFileSchema::Deserialize(data, this);
  }


  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    return nullptr;
  }
  CsvFileSchema* fileSchema() {
    return this;
  }

};

struct ParquetFileSchema {
  std::string path;
  std::vector<int> rowGroupIndices;
  std::vector<int> columnIndices;

  static flatbuffers::Offset<blazingdb::protocol::io::ParquetFile> Serialize(flatbuffers::FlatBufferBuilder &builder, ParquetFileSchema &data) {
      //@todo
      // copy rowGroupIndices and columnIndices
      // make sure you can use these data!
    return blazingdb::protocol::io::CreateParquetFileDirect(builder, data.path.c_str());
  }
  static void Deserialize (const blazingdb::protocol::io::ParquetFile *pointer, ParquetFileSchema* schema){
      schema->path =  std::string{pointer->path()->c_str()};

      //@todo
      // copy rowGroupIndices and columnIndices
      // make sure you can use these data!
  }
};

class LoadParquetFileRequestMessage : public IMessage, ParquetFileSchema {
public:
  LoadParquetFileRequestMessage(std::string path, std::vector<int> rowGroupIndices, std::vector<int> columnIndices)
    : IMessage{}
  {
    this->path = path;
    this->rowGroupIndices = rowGroupIndices;
    this->columnIndices = columnIndices;
  }

  LoadParquetFileRequestMessage(const uint8_t *buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::io::ParquetFile>(buffer);
    ParquetFileSchema::Deserialize(pointer, this);
  }


  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    return nullptr;
  }

  ParquetFileSchema* fileSchema() {
    return this;
  }
};

struct CommunicationNodeSchema {
  std::vector<std::int8_t> buffer;
};

struct CommunicationContextSchema {
  std::vector<CommunicationNodeSchema> nodes;
  std::int32_t masterIndex;
  std::uint64_t token;
};

struct FileSystemBlazingTableSchema {
  std::string name;
  blazingdb::protocol::io::FileSchemaType schemaType;
  CsvFileSchema csv;
  ParquetFileSchema parquet;
  std::vector<std::string> files;
  std::vector<std::string> columnNames;
};

struct FileSystemTableGroupSchema {
  std::vector<FileSystemBlazingTableSchema> tables;
  std::string name;
};


class FileSystemDMLRequestMessage : public IMessage {
public:
  FileSystemDMLRequestMessage(const uint8_t *buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::io::FileSystemDMLRequest>(buffer);
    statement_ =  std::string{pointer->statement()->c_str()};

    auto get_table_group = [] (const blazingdb::protocol::io::FileSystemTableGroup * tableGroup) {
      std::string name = std::string{tableGroup->name()->c_str()};
      std::vector<FileSystemBlazingTableSchema> tables;


      auto _ListFrom = [](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>> *rawNames) {
        std::vector<std::string> columnNames;
        for (const auto& rawName : *rawNames){
          auto name = std::string{rawName->c_str()};
          columnNames.push_back(name);
        }
        return columnNames;
      };

      auto rawTables = tableGroup->tables();
      for (const auto& table : *rawTables) {

        std::string name = std::string{table->name()->c_str()};
        blazingdb::protocol::io::FileSchemaType schemaType = table->schemaType();
        std::vector<std::string> files = _ListFrom(table->files());
        std::vector<std::string> columnNames = _ListFrom(table->columnNames());
        if (schemaType == blazingdb::protocol::io::FileSchemaType::FileSchemaType_CSV) {
          CsvFileSchema csv;
          CsvFileSchema::Deserialize(table->csv(), &csv);

          tables.push_back(FileSystemBlazingTableSchema{
              .name = name,
              .schemaType = schemaType,
              .csv = csv,
              .parquet = ParquetFileSchema{},
              .files = files,
              .columnNames = columnNames,
          });
        } else {
          ParquetFileSchema parquet;
          ParquetFileSchema::Deserialize(table->parquet(), &parquet);

          tables.push_back(FileSystemBlazingTableSchema{
              .name = name,
              .schemaType = schemaType,
              .csv = CsvFileSchema{},
              .parquet = parquet,
              .files = files,
              .columnNames = columnNames,
          });
        }
      }
      return FileSystemTableGroupSchema {
          .tables = tables,
          .name = name,
      };
    };
    tableGroup_ = get_table_group(pointer->tableGroup());

    flatbuffers::unique_ptr<blazingdb::protocol::io::FileSystemDMLRequestT>
        fileSystemDMLRequest = flatbuffers::unique_ptr<
            blazingdb::protocol::io::FileSystemDMLRequestT>(
            flatbuffers::GetRoot<blazingdb::protocol::io::FileSystemDMLRequest>(
                buffer)
                ->UnPack());

    if (fileSystemDMLRequest->communicationContext) {
      const std::unique_ptr<blazingdb::protocol::io::CommunicationContextT>
          &communicationContext = fileSystemDMLRequest->communicationContext;

      std::vector<CommunicationNodeSchema> nodes;
      nodes.reserve(communicationContext->nodes.size());
      std::transform(
          communicationContext->nodes.cbegin(),
          communicationContext->nodes.cend(), std::back_inserter(nodes),
          [](const std::unique_ptr<blazingdb::protocol::io::CommunicationNodeT>
                 &node) { return CommunicationNodeSchema{node->buffer}; });
      communicationContext_ =
          CommunicationContextSchema{nodes, communicationContext->masterIndex,
                               communicationContext->token};
    }
  }

  FileSystemDMLRequestMessage(std::string statement,
                              FileSystemTableGroupSchema tableGroup,
                              const CommunicationContextSchema& communicationContext)
      : statement_{statement},
        tableGroup_{tableGroup},
        communicationContext_{communicationContext},
        IMessage() {}

  flatbuffers::Offset<blazingdb::protocol::io::FileSystemTableGroup> _BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                        FileSystemTableGroupSchema tableGroup) const {
    auto tableNameOffset = builder.CreateString(tableGroup.name);
    std::vector<flatbuffers::Offset<blazingdb::protocol::io::FileSystemBlazingTable>> tablesOffset;

    for (FileSystemBlazingTableSchema& table : tableGroup.tables) {
      auto columnNames = BuildeFlatStringList(builder, table.columnNames);
      auto filesNames = BuildeFlatStringList(builder, table.files);
      flatbuffers::Offset<flatbuffers::String> nameOffset = builder.CreateString(table.name);
      blazingdb::protocol::io::FileSchemaType schemaType = table.schemaType;
      flatbuffers::Offset<blazingdb::protocol::io::CsvFile> csvOffset = CsvFileSchema::Serialize(builder, table.csv);
      flatbuffers::Offset<blazingdb::protocol::io::ParquetFile> parquetOffset = ParquetFileSchema::Serialize(builder, table.parquet);
      flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> filesOffset = builder.CreateVector(filesNames);
      flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> columnNamesOffset = builder.CreateVector(columnNames);
      tablesOffset.push_back( blazingdb::protocol::io::CreateFileSystemBlazingTable(builder, nameOffset, schemaType, csvOffset, parquetOffset, filesOffset, columnNamesOffset));
    }

    auto tables = builder.CreateVector(tablesOffset);
    return blazingdb::protocol::io::CreateFileSystemTableGroup(builder, tables, tableNameOffset);
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
       flatbuffers::FlatBufferBuilder builder;
    auto logicalPlan_offset = builder.CreateString(statement_);
    auto tableGroupOffset = _BuildTableGroup(builder, tableGroup_);

    std::vector<flatbuffers::Offset<blazingdb::protocol::io::CommunicationNode>>
        nodeOffsets;
    nodeOffsets.resize(communicationContext_.nodes.size());
    std::transform(
        communicationContext_.nodes.cbegin(),
        communicationContext_.nodes.cend(), nodeOffsets.begin(),
        [&builder](const CommunicationNodeSchema &node) {
          return blazingdb::protocol::io::CreateCommunicationNodeDirect(
              builder, &node.buffer);
        });

    auto communicationContextOffset =
        blazingdb::protocol::io::CreateCommunicationContextDirect(
            builder, &nodeOffsets, communicationContext_.masterIndex, communicationContext_.token);

    builder.Finish(blazingdb::protocol::io::CreateFileSystemDMLRequest(builder, logicalPlan_offset, tableGroupOffset, communicationContextOffset));
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  const std::string &statement() const noexcept { return statement_; }
  const FileSystemTableGroupSchema &tableGroup() const noexcept { return tableGroup_; }
  const CommunicationContextSchema &communicationContext() const noexcept { return communicationContext_; }

private:
  std::string statement_;
  FileSystemTableGroupSchema tableGroup_;
  CommunicationContextSchema communicationContext_;
};


}  // namespace io
}  // namespace message
}  // namespace blazingdb
