import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.transport.channel import ResponseSchema
from blazingdb.protocol.transport.channel import MakeRequestBuffer
from blazingdb.protocol.orchestrator import DMLResponseSchema
from blazingdb.protocol.interpreter import GetResultRequestSchema


def main():
    client = blazingdb.protocol.ZeroMqClient('ipc:///tmp/socket')
    responseBuffer = client.send('hola desde python')
    print(responseBuffer)
    # responseBuffer = client.send(requestBuffer)
    # print(responseBuffer)

if __name__ == '__main__':
    main()
