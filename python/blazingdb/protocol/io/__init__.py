import flatbuffers
import copy
import numpy
from blazingdb.messages.blazingdb.protocol.io \
    import FileSystemRegisterRequest, FileSystemDeregisterRequest, HDFS, S3, POSIX

from blazingdb.messages.blazingdb.protocol.io import DriverType, EncryptionType, FileSystemConnection

DriverType = DriverType.DriverType
EncryptionType = EncryptionType.EncryptionType
FileSystemType = FileSystemConnection.FileSystemConnection

class FileSystemRegisterRequestSchema:    
    def __init__(self, authority, root, type, params):
        self.authority = authority
        self.root = root
        self.params = params
        self.type = type
  
    def ToBuffer(self):
        builder = flatbuffers.Builder(1024)
        authority = builder.CreateString(self.authority)
        root = builder.CreateString(self.root)
        if self.type == FileSystemType.HDFS:
            fileSystemConnection, fileSystemConnectionType = MakeHdfsFileSystemConnection(builder, self.params)
        elif self.type == FileSystemType.S3:
            fileSystemConnection, fileSystemConnectionType = MakeS3FileSystemRegisterRequest(builder, self.params)
        else:
            fileSystemConnection, fileSystemConnectionType = MakePosixFileSystemConnection(builder, self.params)

        FileSystemRegisterRequest.FileSystemRegisterRequestStart(builder)
        FileSystemRegisterRequest.FileSystemRegisterRequestAddAuthority(builder, authority)
        FileSystemRegisterRequest.FileSystemRegisterRequestAddRoot(builder, root)

        FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnectionType(builder, fileSystemConnectionType)
        FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnection(builder, fileSystemConnection)
        fs = FileSystemRegisterRequest.FileSystemRegisterRequestEnd(builder)
        builder.Finish(fs)
        return builder.Output()

class FileSystemDeregisterRequestSchema: 
    def __init__(self, authority):
        self.authority = authority
        
    def ToBuffer(self):
        builder = flatbuffers.Builder(1024)
        authority = builder.CreateString(self.authority)
        FileSystemDeregisterRequest.FileSystemDeregisterRequestStart(builder)
        FileSystemDeregisterRequest.FileSystemDeregisterRequestAddAuthority(builder, authority)
        fs = FileSystemDeregisterRequest.FileSystemDeregisterRequestEnd(builder)
        builder.Finish(fs)
        return builder.Output()
 


def MakePosixFileSystemConnection(builder, params):
    return 0, FileSystemType.POSIX

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
    return paramObj, FileSystemType.HDFS


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
    return paramObj, FileSystemType.S3
