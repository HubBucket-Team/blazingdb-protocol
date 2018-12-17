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

        FileSystemRegisterRequest.FileSystemRegisterRequestAddFileSystemConnectionType(builder,
                                                                                       fileSystemConnectionType)
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

# # schema.kwargs
# def _CreateFileSchema(builder, schema):
#     if schema.type == "csv":
#         type = FileSchemaType.CsvFile
#         obj = CsvFileSchema(**schema.kwargs)
#     else:
#         type = FileSchemaType.ParquetFile
#         obj = ParquetFileSchema(**schema.kwargs)
#     return obj._allocate_segments(builder), type
#
# def _CreateFiles (builder, files):
#     None
#
# def _CreateColumnNames(builder, column_names):
#     None
#
# # schema.table_name
# # schema.column_names
#
# def _CreateTable(builder, schema, files):
#     assert len(schema.table_name) > 0
#     table_name = builder.CreateString(schema.table_name)
#
#     FileSystemBlazingTable.FileSystemBlazingTableStart(builder)
#     FileSystemBlazingTable.FileSystemBlazingTableAddName(table_name)
#
#     fileConfig, fileConfigType = _CreateFileSchema(builder, schema)
#     files = _CreateFiles(builder, files)
#     columnNames = _CreateColumnNames(builder, schema.column_names)
#     FileSystemBlazingTable.FileSystemBlazingTableAddFileConfigType(fileConfigType)
#     FileSystemBlazingTable.FileSystemBlazingTableAddFileConfig(fileConfig)
#     FileSystemBlazingTable.FileSystemBlazingTableAddFiles(files)
#     FileSystemBlazingTable.FileSystemBlazingTableAddColumnNames(columnNames)
#     return FileSystemBlazingTable.FileSystemBlazingTableEnd(builder)
#
# def _CreateTables(builder, sql_data):
#     FileSystemTableGroup.FileSystemTableGroupStartTablesVector(builder, len(sql_data.items()))
#     for schema, files in sql_data.items():
#         table = _CreateTable(builder, schema, files)
#         builder.PrependUOffsetTRelative(table)
#     return FileSystemTableGroup.FileSystemTableGroupEnd(builder)
#
# def _CreateTableGroup(builder, sql_data):
#     dbname = builder.CreateString('main')
#     tables = _CreateTables(builder, sql_data)
#     FileSystemTableGroup.FileSystemTableGroupStart(builder)
#     FileSystemTableGroup.FileSystemTableGroupAddName(dbname)
#     FileSystemTableGroup.FileSystemTableGroupAddTables(tables)
#     return FileSystemTableGroup.FileSystemTableGroupEnd(builder)
#
# class FileSystemDMLRequestSchema:
#     def __init__(self, statement, sql_data):
#         self.statement = statement
#         self.sql_data = sql_data
#
#     def ToBuffer(self):
#         builder = flatbuffers.Builder(1024)
#         statement = builder.CreateString(self.statement)
#         tableGroup = _CreateTableGroup(builder, self.sql_data)
#         FileSystemDMLRequest.FileSystemDMLRequestStart(builder)
#         FileSystemDMLRequest.FileSystemDMLRequestAddStatement(builder, statement)
#         FileSystemDMLRequest.FileSystemDMLRequestAddTableGroup(tableGroup)
#         fs = FileSystemDMLRequest.FileSystemDMLRequestEnd(builder)
#         builder.Finish(fs)
#         return builder.Output()
