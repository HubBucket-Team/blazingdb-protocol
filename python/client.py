import blazingdb.protocol
import blazingdb.protocol.requests


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  client = blazingdb.protocol.Client(connection)

  requestBuffer = blazingdb.protocol.requests.MakeDMLRequest('Select')

  responseBuffer = client.send(requestBuffer)

  print(responseBuffer)


if __name__ == '__main__':
  main()
