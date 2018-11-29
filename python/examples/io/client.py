import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.transport.channel import ResponseSchema
from blazingdb.protocol.transport.channel import MakeRequestBuffer
from blazingdb.protocol.orchestrator import DMLResponseSchema
from blazingdb.protocol.interpreter import GetResultRequestSchema

import blazingdb.protocol.io  as io
from collections import namedtuple


def main():
    ACCESS_TOKEN = 456
    connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
    client = blazingdb.protocol.Client(connection)
    d = {
        'host': 'string',
        'port': 8080,
        'user': 'string',
        'driverType': io.DriverType.DriverType.LIBHDFS,
        'kerberosTicket': 'string'
    }
    hdfs = namedtuple("HDFS", d.keys())(*d.values())
    print (hdfs)

    buffer = io.MakeFileSystemRegisterRequest('authority_name', 'root/', io.FileSystemConnection.FileSystemConnection.HDFS, hdfs)
    print(client.send(buffer))


if __name__ == '__main__':
    main()
