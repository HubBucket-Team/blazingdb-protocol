import abc
import os
import random
import socket
import struct
import asyncio
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

    def connect_sync(self):
        try:
            self.socket_.connect(self.address_)
        except Exception as e:
            raise RuntimeError("Communication error connecting to {0}: {1}".format(self.address_, str(e)))

    async def connect_async(self):
        try:
            return await asyncio.open_unix_connection(path=self.address_)
            return (reader, writer)
        except Exception as e:
            raise RuntimeError("Communication error connecting to {0}: {1}".format(self.address_, str(e)))


class Client:

    def __init__(self, connection):
        self.connection_ = connection
        connection.connect_sync()

    def send(self, _buffer):
        try:
            length = struct.pack('I', len(_buffer))
            self.connection_.socket_.sendall(length)
            self.connection_.socket_.sendall(_buffer)
            length = struct.unpack('I', self.connection_.socket_.recv(4))[0]
            return self.connection_.socket_.recv(length)
        except Exception as e:
            raise RuntimeError("Communication error sending data to {0}: {1}".format(self.connection_.address(), str(e)))

class AsyncClient:

    def __init__(self, connection):
        self.connection_ = connection

    async def send(self, request_buffer):
        reader, writer = await self.connection_.connect_async()
        try:
            request_length = struct.pack('I', len(request_buffer))
            writer.write(request_length)
            writer.write(request_buffer)
            writer.write_eof()
            await writer.drain()
            response_length = struct.unpack('I', await reader.readexactly(4))[0]
            response_buffer = await reader.readexactly(response_length)
            while not reader.at_eof():
                await reader.read()
            return response_buffer
        except Exception as e:
            raise RuntimeError("Communication error sending data to {0}: {1}".format(self.connection_.address(), str(e)))
