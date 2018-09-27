import blazingdb.protocol


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  client = blazingdb.protocol.Client(connection)

  responseBuffer = client.send(b'BlazingDB Request')

  print(responseBuffer)


if __name__ == '__main__':
  main()
