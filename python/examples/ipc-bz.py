# Sample source code from the Tutorial Introduction in the documentation.
import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel
import numpy
import multiprocessing as mp
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray


def create_sample_device_data():
  a = numpy.random.randn(1, 32)
  a = a.astype(numpy.int8)
  print('orig: ', a)
  print('orig: ', a.shape)

  a_gpu = drv.mem_alloc(a.size * a.dtype.itemsize)
  drv.memcpy_htod(a_gpu, a)
  return a_gpu

unix_path = '/tmp/demo.socket'

def client():
  drv.init()
  dev = drv.Device(0)
  ctx_gpu = dev.make_context()
  connection = blazingdb.protocol.UnixSocketConnection(unix_path)
  sock = blazingdb.protocol.Client(connection)

  x_gpu = create_sample_device_data()
  h = drv.mem_get_ipc_handle(x_gpu)

  print('send handler')
  print(h)

  res = sock.send(bytes(h))
  print(res)
  ctx_gpu.pop()


def server():
  print('waiting')

  connection = blazingdb.protocol.UnixSocketConnection(unix_path)
  server = blazingdb.protocol.Server(connection)

  def controller(h):
    drv.init()
    dev = drv.Device(0)
    ctx_gpu = dev.make_context()

    print('receive handler')
    print(bytearray(bytes(h)))
    x_ptr = drv.IPCMemoryHandle(bytearray(bytes(h)))
    x_gpu = gpuarray.GPUArray((1, 32), numpy.int8, gpudata=x_ptr)
    print('gpu:  ', x_gpu.get())
    ctx_gpu.pop()
    return b'hi back!'

  server.handle(controller)

import time

if __name__ == '__main__':
  p1 = mp.Process(target=client)
  p2 = mp.Process(target=server)
  # p2.start()
  # time.sleep(0.5)
  p1.start()

