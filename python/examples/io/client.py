import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel

from blazingdb.protocol.io import FileSystemRegisterRequestSchema, FileSystemDeregisterRequestSchema, CsvFileSchema
from collections import namedtuple

from blazingdb.protocol.io import DriverType, FileSystemType


def main():
    connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
    client = blazingdb.protocol.Client(connection)
    
    d = {
        'host': 'string',
        'port': 8080,
        'user': 'string',
        'driverType': DriverType.LIBHDFS,
        'kerberosTicket': 'string'
    }
    hdfs = namedtuple("HDFS", d.keys())(*d.values())
    print(hdfs)

    schema = FileSystemRegisterRequestSchema('authority_name', 'root/', FileSystemType.HDFS, hdfs)
    # schema = FileSystemDeregisterRequestSchema(authority = 'authority_name_de')
    print(client.send(schema.ToBuffer()))


if __name__ == '__main__':
    # main()
    path = 'path'
    delimiter = '|'
    line_terminator = '\n'
    skip_rows = 0
    names = ['1', '2']
    dtypes = [1,  2]

    requestSchema = blazingdb.protocol.io.CsvFileSchema(path=path, delimiter=delimiter, lineTerminator=line_terminator,
                                                        skipRows=skip_rows, names=names, dtypes=dtypes)
    buffer = requestSchema.ToBuffer()
    obj = blazingdb.protocol.io.CsvFileSchema.From(buffer)

    print(obj.path)
    print(obj.names)
    print(obj.dtypes)