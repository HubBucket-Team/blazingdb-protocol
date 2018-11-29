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

enum FileSystemType {
  FileSystemType_NONE = 0,
  FileSystemType_POSIX = 1,
  FileSystemType_HDFS = 2,
  FileSystemType_S3 = 3,
};

using namespace blazingdb::protocol;

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
  ~FileSystemRegisterRequestMessage() = default;

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

  std::string get_authority() const {
    return authority;
  }
  std::string get_root () const {
    return root;
  }
  

private:
  std::string authority;
  std::string root;
  FileSystemType type;

  // POSIX posix;
  HDFS hdfs;
  S3 s3;
};

}  // namespace io
}  // namespace message
}  // namespace blazingdb
