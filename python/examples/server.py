import blazingdb.protocol
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  server = blazingdb.protocol.Server(connection)

  def controller(requestBuffer):
    request = blazingdb.protocol.transport.channel.RequestSchemaFrom(
      requestBuffer)

    print(request.header)

    dml = blazingdb.protocol.orchestrator.DMLRequestSchema.From(request.payload)

    print(dml.query)

    dmlResponse = blazingdb.protocol.orchestrator.DMLResponseSchema(
      resultToken='T-O-K-E-N')

    responseBuffer = \
      blazingdb.protocol.transport.channel.ResponseSchema(
        status=blazingdb.protocol.transport.channel.Status.Success,
        payload=dmlResponse.ToBuffer()
      ).ToBuffer()

    return responseBuffer

  server.handle(controller)


if __name__ == '__main__':
  main()

