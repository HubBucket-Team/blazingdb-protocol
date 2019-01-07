import blazingdb.protocol
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel

from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.protocol.transport.channel import ResponseSchema
from blazingdb.protocol.transport.channel import RequestSchemaFrom
from blazingdb.protocol.orchestrator import DMLResponseSchema, NodeConnectionSchema


def main():
    connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
    server = blazingdb.protocol.Server(connection)

    def controller(requestBuffer):
        request = RequestSchemaFrom(requestBuffer)
        print(request.header)
        dmlResponse = DMLResponseSchema(resultToken=123456, nodeConnection=NodeConnectionSchema(
            path='/',
            type=1,
        ), calciteTime=1)
        responseBuffer = ResponseSchema(
            status=Status.Success,
            payload=dmlResponse.ToBuffer()
        ).ToBuffer()
        return responseBuffer

    server.handle(controller)


if __name__ == '__main__':
    main()
