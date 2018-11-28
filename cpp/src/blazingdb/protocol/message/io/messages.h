#pragma once

#include <string>
#include <blazingdb/protocol/api.h>
#include "flatbuffers/flatbuffers.h"
#include "../messages.h"

using namespace blazingdb::protocol;

namespace blazingdb {
namespace messages {
namespace io {

struct POSIX {

};

enum DriverType {
    UNDEFINED,
    LIBHDFS, // LIBHDFS requires Java CLASSPATH and native HDFS in LD_LIBRARY_PATH
    LIBHDFS3 // LIBHDFS3 requires C++ pivotalrd-libhdfs3 library in LD_LIBRARY_PATH
};

struct HDFS {
    std::string host;
    int port;
    std::string user;
    DriverType driverType;
    std::string kerberosTicket;
};

enum EncryptionType {
    UNDEFINED,
    NONE,
    AES_256,
    AWS_KMS // AWS Key Management Service
};

struct S3 {
    std::string bucketName;
    EncryptionType encryptionType;
    std::string kmsKeyAmazonResourceName;
    std::string accessKeyId;
    std::string secretKey;
    std::string sessionToken;
};

union FileSystemParams {
    POSIX posix;
    HDFS hdfs;
    S3 s3; 
};

class FileSystemRegisterRequestMessage : public IMessage {
public:

  FileSystemRegisterRequestMessage(const std::string &root, blazingdb::protocol::io::FileSystemType type, const FileSystemParams &params)
      : root{root}, type{type}, params{params}, IMessage()
  {

  }
  FileSystemRegisterRequestMessage (const uint8_t* buffer)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::io::FileSystemRegisterRequest>(buffer);
    root = std::string{pointer->root()->c_str()};
    type = pointer->type();
    // params = pointer->params();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder; 
    blazingdb::protocol::io::FileSystemParams paramType;
    if (type == blazingdb::protocol::io::FileSystemType::FileSystemType_HDFS) {
      paramType = blazingdb::protocol::io::FileSystemParams::FileSystemParams_HDFS;
      auto union_obj = blazingdb::protocol::io::CreateHDFSDirect(builder, params.hdfs.host.c_str(). params.hdfs.port, params.hdfs.user.c_str(), params.hdfs.driverType, params.hdfs.kerberosTicket.c_str());
      builder.Finish(CreateFileSystemRegisterRequestDirect(builder, type, root.c_str(), paramType, union_obj.Union() ));
    }
    else if (type == blazingdb::protocol::io::FileSystemType::FileSystemType_S3) {
      paramType = blazingdb::protocol::io::FileSystemParams::FileSystemParams_S3;
      auto union_obj = blazingdb::protocol::io::CreateS3Direct(builder, params.s3.bucketName.c_str(). params.s3.EncryptionType, params.s3.kmsKeyAmazonResourceName.c_str(), params.s3.accessKeyId.c_str(), params.s3.secretKey.c_str(), params.s3.sessionToken.c_str());
      builder.Finish(CreateFileSystemRegisterRequestDirect(builder, type, root.c_str(), paramType, 0 ));
    }
    else {
      paramType = blazingdb::protocol::io::FileSystemParams::FileSystemParams_POSIX;
      builder.Finish(CreateFileSystemRegisterRequestDirect(builder, type, root.c_str(), paramType, 0 ));
    }
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

private:
  std::string root;
  blazingdb::protocol::io::FileSystemType type;
  FileSystemParams params;
};

}
}
}
