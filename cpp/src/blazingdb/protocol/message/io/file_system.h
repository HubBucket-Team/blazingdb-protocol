#ifndef BLAZINGDB_PROTOCOL_MESSAGE_IO_FILE_SYSTEM_H
#define BLAZINGDB_PROTOCOL_MESSAGE_IO_FILE_SYSTEM_H
 
#include <blazingdb/protocol/message/messages.h>
#include <blazingdb/protocol/message/interpreter/utils.h>

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
  FileSystemDeregisterRequestMessage(const std::string& string_value);

  FileSystemDeregisterRequestMessage (const uint8_t* buffer);

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override;

  std::string getAuthority ();
};


class FileSystemRegisterRequestMessage : public IMessage {
public:
  FileSystemRegisterRequestMessage(const std::string &authority, const std::string &root);

  FileSystemRegisterRequestMessage(const std::string &authority, const std::string &root, const HDFS &hdfs);

  FileSystemRegisterRequestMessage(const std::string &authority, const std::string &root, const S3 &s3);

  FileSystemRegisterRequestMessage(const uint8_t *buffer);

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override;

  std::string getAuthority() const;
  std::string getRoot () const;
  
  HDFS getHdfs() const;

  S3 getS3() const;
  
  bool isLocal() const;

  bool isHdfs() const;

  bool isS3() const;
  

private:
  std::string authority;
  std::string root;
  FileSystemType type;

  // POSIX posix;
  HDFS hdfs;
  S3 s3;
};


std::vector<flatbuffers::Offset<flatbuffers::String>>  BuildeFlatStringList(flatbuffers::FlatBufferBuilder &builder, const std::vector<std::string> &strings);

struct CsvFileSchema {
  std::string path;
  std::string delimiter = "|";
  std::string line_terminator = "\n";
  int skip_rows = 0;
  std::vector<std::string> names;
  std::vector<int> dtypes;


  static flatbuffers::Offset<blazingdb::protocol::io::CsvFile> Serialize(flatbuffers::FlatBufferBuilder &builder, CsvFileSchema data);

  static void Deserialize(const blazingdb::protocol::io::CsvFile *pointer, CsvFileSchema* schema);
};

class LoadCsvFileRequestMessage : public IMessage, CsvFileSchema {
public:
  LoadCsvFileRequestMessage(const std::string path,
      const std::string & delimiter,
			const std::string & line_terminator,
			int skip_rows,
			const std::vector<std::string> & names,
			const std::vector<int> & dtypes);

  LoadCsvFileRequestMessage(const uint8_t *buffer) ;

  
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override ;
  CsvFileSchema* fileSchema() ;

};

struct ParquetFileSchema {
  std::string path;
  std::vector<int> rowGroupIndices;
  std::vector<int> columnIndices;

  static flatbuffers::Offset<blazingdb::protocol::io::ParquetFile> Serialize(flatbuffers::FlatBufferBuilder &builder, ParquetFileSchema &data) ;
  static void Deserialize (const blazingdb::protocol::io::ParquetFile *pointer, ParquetFileSchema* schema);
};

class LoadParquetFileRequestMessage : public IMessage, ParquetFileSchema {
public:
  LoadParquetFileRequestMessage(std::string path, std::vector<int> rowGroupIndices, std::vector<int> columnIndices);

  LoadParquetFileRequestMessage(const uint8_t *buffer);


  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override;

  ParquetFileSchema* fileSchema() ;
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
  std::string name; //ok
  blazingdb::protocol::FileSchemaType schemaType; //ok
  CsvFileSchema csv; //deprecated
  ParquetFileSchema parquet; //deprecated
  blazingdb::protocol::BlazingTableSchema gdf; //ok
  blazingdb::protocol::TableSchemaSTL tableSchema; //ok

  std::vector<std::string> columnNames{};
  std::vector<std::string> columnTypes{};
  
};

struct FileSystemTableGroupSchema {
  std::vector<FileSystemBlazingTableSchema> tables;
  std::string name;
};


class FileSystemDMLRequestMessage : public IMessage {
public: 
  FileSystemDMLRequestMessage(const uint8_t *buffer);
  FileSystemDMLRequestMessage(std::string statement, FileSystemTableGroupSchema tableGroup,  	                              
                                  const CommunicationContextSchema &communicationContext,
                                  uint64_t resultToken);

  flatbuffers::Offset<blazingdb::protocol::io::FileSystemTableGroup> _BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                        FileSystemTableGroupSchema tableGroup) const ;

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override ;

  const std::string &statement() const noexcept;
  const FileSystemTableGroupSchema &tableGroup() const noexcept;
  const CommunicationContextSchema &communicationContext() const noexcept;
  const uint64_t &resultToken() const noexcept;

private:
  std::string statement_;
  FileSystemTableGroupSchema tableGroup_;
  CommunicationContextSchema communicationContext_;
  uint64_t resultToken_;
};


}  // namespace io
}  // namespace message
}  // namespace blazingdb

#endif // BLAZINGDB_PROTOCOL_MESSAGE_IO_FILE_SYSTEM_H
