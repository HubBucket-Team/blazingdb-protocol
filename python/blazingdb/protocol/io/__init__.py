import flatbuffers
import copy
import numpy
from blazingdb.messages.blazingdb.protocol.io \
    import FileSystemRegisterRequest, FileSystemConnection, FileSystemType, DriverType, EncryptionType, HDFS, S3, POSIX

import blazingdb.protocol.transport
import blazingdb.protocol.transport as transport
from blazingdb.messages.blazingdb.protocol.Status import Status

# class FileSystemRegisterRequestSchema(transport.schema(FileSystemRegisterRequest)):
#   fsType = transport.NumberSegment()
#   root = transport.StringSegment()
#   params = transport.UnionSegment(FileSystemParams)


class FileSystemRegisterRequestSchema:
  def __init__(self, authority, root, params):
      self.authority = authority
      self.root = root
      self.params = params

def MakeFileSystemRegisterRequest(authority, root, type, params):
    builder = flatbuffers.Builder(1024)
    authority = builder.CreateString(authority)
    root = builder.CreateString(root)
    if type == FileSystemConnection.FileSystemConnection.HDFS:
        fileSystemConnection, fileSystemConnectionType = MakeHdfsFileSystemConnection(builder, params)
    elif type == FileSystemConnection.FileSystemConnection.S3:
        fileSystemConnection, fileSystemConnectionType = MakeS3FileSystemRegisterRequest(builder, params)
    else:
        fileSystemConnection, fileSystemConnectionType = MakePosixFileSystemConnection(builder, params)

    FileSystemRegisterRequest.FileSystemRegisterRequestStart(builder)
    FileSystemRegisterRequest.FileSystemRegisterRequestAddAuthority(builder, authority)
    FileSystemRegisterRequest.FileSystemRegisterRequestAddRoot(builder, root)

    FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnectionType(builder, fileSystemConnectionType)
    FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnection(builder, fileSystemConnection)
    fs = FileSystemRegisterRequest.FileSystemRegisterRequestEnd(builder)
    builder.Finish(fs)
    return builder.Output()


def MakePosixFileSystemConnection(builder, params):
    return None, FileSystemConnection.FileSystemConnection.POSIX

def MakeHdfsFileSystemConnection(builder, params):
    host = builder.CreateString(params.host)
    user = builder.CreateString(params.user)
    ticket = builder.CreateString(params.kerberosTicket)
    HDFS.HDFSStart(builder)
    HDFS.HDFSAddHost(builder, host)
    HDFS.HDFSAddPort(builder, params.port)
    HDFS.HDFSAddUser(builder, user)
    HDFS.HDFSAddDriverType(builder, params.driverType)  # check if it is enum
    HDFS.HDFSAddKerberosTicket(builder, ticket)
    paramObj = HDFS.HDFSEnd(builder)
    return paramObj, FileSystemConnection.FileSystemConnection.HDFS


def MakeS3FileSystemRegisterRequest(builder, params):
    bucketName = builder.CreateString(params.bucketName)
    kmsKeyAmazonResourceName = builder.CreateString(params.kmsKeyAmazonResourceName)
    accessKeyId = builder.CreateString(params.accessKeyId)
    secretKey = builder.CreateString(params.secretKey)
    sessionToken = builder.CreateString(params.sessionToken)
    S3.S3Start(builder)
    S3.S3AddBucketName(builder, bucketName)
    S3.S3AddEncryptionType(builder, params.encryptionType) # check if it is enum
    S3.S3AddKmsKeyAmazonResourceName(builder, kmsKeyAmazonResourceName)
    S3.S3AddAccessKeyId(builder, accessKeyId)
    S3.S3AddSecretKey(builder, secretKey)
    S3.S3AddSessionToken(builder, sessionToken)
    paramObj = S3.S3End(builder)
    return paramObj, FileSystemConnection.FileSystemConnection.S3
