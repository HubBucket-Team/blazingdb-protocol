import blazingdb.protocol


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  server = blazingdb.protocol.Server(connection)

  def controller(requestBuffer):
    print(requestBuffer)
    responseBuffer = b'BlazingDB Response'
    return responseBuffer

  server.handle(controller)


if __name__ == '__main__':
  main()

