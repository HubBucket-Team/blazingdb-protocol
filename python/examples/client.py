import blazingdb.protocol
import blazingdb.protocol.orchestrator


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  client = blazingdb.protocol.Client(connection)

  requestBuffer = blazingdb.protocol.orchestrator.MakeDMLRequest(
      'select * from Table')

  responseBuffer = client.send(requestBuffer)

  response = blazingdb.protocol.orchestrator.DMLResponseFrom(responseBuffer)

  print(response.status)


if __name__ == '__main__':
  main()
