import flatbuffers
import copy
import numpy
import blazingdb.protocol.transport as transport
from blazingdb.messages.blazingdb.protocol.io \
    import FileSystemRegisterRequest, FileSystemDeregisterRequest, HDFS, S3, POSIX, CsvFile, ParquetFile

from blazingdb.messages.blazingdb.protocol.io import DriverType, EncryptionType, FileSystemConnection, FileSchemaType

from blazingdb.messages.blazingdb.protocol.io import FileSystemDMLRequest, FileSystemTableGroup, FileSystemBlazingTable

DriverType = DriverType.DriverType
EncryptionType = EncryptionType.EncryptionType
FileSystemType = FileSystemConnection.FileSystemConnection
FileSchemaType = FileSchemaType.FileSchemaType

#todo, crear mappers para unions, see union_segment
class FileSystemRegisterRequestSchema:
    def __init__(self, authority, root, type, params):
        self.authority = authority
        self.root = root
        self.params = params
        self.type = type

    #todo, crear mappers para unions
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

        FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnectionType(builder,
                                                                                       fileSystemConnectionType)
        FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnection(builder, fileSystemConnection)
        fs = FileSystemRegisterRequest.FileSystemRegisterRequestEnd(builder)
        builder.Finish(fs)
        return builder.Output()


#todo, crear mappers para unions, see union_segment
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
    S3.S3AddEncryptionType(builder, params.encryptionType)  # check if it is enum
    S3.S3AddKmsKeyAmazonResourceName(builder, kmsKeyAmazonResourceName)
    S3.S3AddAccessKeyId(builder, accessKeyId)
    S3.S3AddSecretKey(builder, secretKey)
    S3.S3AddSessionToken(builder, sessionToken)
    paramObj = S3.S3End(builder)
    return paramObj, FileSystemType.S3


class CsvFileSchema(transport.schema(CsvFile)):
    path = transport.StringSegment()
    delimiter = transport.StringSegment()
    lineTerminator = transport.StringSegment()
    skipRows = transport.NumberSegment()
    names = transport.VectorStringSegment(transport.StringSegment)
    dtypes = transport.VectorSegment(transport.NumberSegment)


class ParquetFileSchema(transport.schema(ParquetFile)):
    path = transport.StringSegment()
    rowGroupIndices = transport.VectorSegment(transport.NumberSegment)
    columnIndices = transport.VectorSegment(transport.NumberSegment)


class FileSystemBlazingTableSchema(transport.schema(FileSystemBlazingTable)):
    name = transport.StringSegment()
    schemaType = transport.NumberSegment()
    csv = transport.SchemaSegment(CsvFileSchema)
    parquet = transport.SchemaSegment(ParquetFileSchema)
    files = transport.VectorStringSegment(transport.StringSegment)
    columnNames = transport.VectorStringSegment(transport.StringSegment)


class FileSystemTableGroupSchema(transport.schema(FileSystemTableGroup)):
    tables = transport.VectorSchemaSegment(FileSystemBlazingTableSchema)
    name = transport.StringSegment()


class FileSystemDMLRequestSchema(transport.schema(FileSystemDMLRequest)):
    statement = transport.StringSegment()
    tableGroup = transport.SchemaSegment(FileSystemTableGroupSchema)


def _GetParquetSchema(kwargs):
    path = kwargs.get('path', '')
    rowGroupIndices = kwargs.get('rowGroupIndices', [])
    columnIndices = kwargs.get('columnIndices', [])
    return ParquetFileSchema(path=path, rowGroupIndices=rowGroupIndices, columnIndices=columnIndices)


def _GetCsvSchema(kwargs):
    path =  kwargs.get('path', '')
    delimiter =  kwargs.get('delimiter', '')
    lineTerminator =  kwargs.get('lineTerminator', '')
    skipRows =  kwargs.get('skipRows', 0)
    names =  kwargs.get('names', [])
    dtypes =  kwargs.get('dtypes', [])
    return CsvFileSchema(path=path, delimiter=delimiter, lineTerminator=lineTerminator, skipRows=skipRows, names=names,
                         dtypes=dtypes)


def BuildFileSystemDMLRequestSchema(statement, tableGroupDto):
    tableGroupName = tableGroupDto['name']
    tables = []
    for index, t in enumerate(tableGroupDto['tables']):
        tableName = t['name']
        columnNames = t['columnNames']
        files = t['files']
        schemaType = t['schemaType']

        if schemaType == FileSchemaType.PARQUET:
            parquet = _GetParquetSchema(t['parquet'])
            csv = _GetCsvSchema({})
        else:
            csv = _GetCsvSchema(t['csv'])
            parquet = _GetParquetSchema({})

        table = FileSystemBlazingTableSchema(name=tableName, schemaType=schemaType, parquet=parquet, csv=csv,
                                             files=files, columnNames=columnNames)

        tables.append(table)
    tableGroup = FileSystemTableGroupSchema(tables=tables, name=tableGroupName)
    return FileSystemDMLRequestSchema(statement=statement, tableGroup=tableGroup)
 