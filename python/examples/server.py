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

    responseBuffer = \
      blazingdb.protocol.orchestrator.MakeDMLResponse('t-o-k-e-n')

    return responseBuffer

  server.handle(controller)


if __name__ == '__main__':
  main()

