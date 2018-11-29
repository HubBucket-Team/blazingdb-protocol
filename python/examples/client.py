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
    ACCESS_TOKEN = 456
    connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
    client = blazingdb.protocol.Client(connection)
    getResult = GetResultRequestSchema(resultToken=123456)
    requestBuffer = MakeRequestBuffer(InterpreterMessage.GetResult, ACCESS_TOKEN, getResult)
    responseBuffer = client.send(requestBuffer)
    response = ResponseSchema.From(responseBuffer)
    print(response.status)

    dmlResponse = DMLResponseSchema.From(response.payload)
    print(dmlResponse.resultToken)
    # responseBuffer = client.send(requestBuffer)
    # print(responseBuffer)

if __name__ == '__main__':
    main()
