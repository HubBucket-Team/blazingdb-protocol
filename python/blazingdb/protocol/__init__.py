import abc
import os
import socket
import struct


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
    address = connection.address()
    if (os.path.exists(address)):
      os.unlink(address)
    connection.socket_.bind(connection.address())
    connection.socket_.listen(4)

  def handle(self, callback):
    while True:
      client, address = self.connection_.socket_.accept()
      with client:
        length = struct.unpack('I', client.recv(4))[0]
        requestBuffer = client.recv(length)
        responseBuffer = callback(requestBuffer)
        client.sendall(struct.pack('I', len(responseBuffer)))
        client.sendall(responseBuffer)


class Client:

  def __init__(self, connection):
    self.connection_ = connection
    try:
      connection.socket_.connect(connection.address())
    except socket.error:
      raise RuntimeError("connect error")

  def send(self, _buffer):
    length = struct.pack('I', len(_buffer))
    self.connection_.socket_.sendall(length)
    self.connection_.socket_.sendall(_buffer)
    length = struct.unpack('I', self.connection_.socket_.recv(4))[0]
    return self.connection_.socket_.recv(length)
