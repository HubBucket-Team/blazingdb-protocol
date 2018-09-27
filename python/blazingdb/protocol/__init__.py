import abc
import os
import socket


__all__ = ['UnixSocketConnection', 'Server', 'Client']


class Buffer(abc.ABC):

  def __len__(self):
    return NotImplemented

Buffer.register(bytes)


class UnixSocketConnection:

  def __init__(self, path):
    self.address_ = path
    self.socket_ = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)

  def __del__(self):
    self.socket_.close()

  def address(self):
    return self.address_


class Server:

  def __init__(self, connection):
    self.connection_ = connection
    os.unlink(connection.address())
    connection.socket_.bind(connection.address())
    connection.socket_.listen(4)

  def handle(self, callback):
    while True:
      client, address = self.connection_.socket_.accept()
      with client:
        requestBuffer = client.recv(4096)
        responseBuffer = callback(requestBuffer)
        client.sendall(responseBuffer)


class Client:

  def __init__(self, connection):
    self.connection_ = connection
    try:
      connection.socket_.connect(connection.address())
    except socket.error:
      raise RuntimeError("connect error")

  def send(self, buffer):
    self.connection_.socket_.sendall(buffer)
    responseBuffer = self.connection_.socket_.recv(4096)
    return responseBuffer

