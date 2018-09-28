import blazingdb.protocol
import blazingdb.protocol.requests


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  server = blazingdb.protocol.Server(connection)

  def controller(requestBuffer):
    blazingdb.protocol.requests.DMLRequestFrom(requestBuffer)

    responseBuffer = b'BlazingDB Response'
    return responseBuffer

  server.handle(controller)


if __name__ == '__main__':
  main()

