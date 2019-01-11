import abc
import os
import random
import socket
import struct
import threading

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
