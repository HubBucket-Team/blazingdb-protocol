# Sample source code from the Tutorial Introduction in the documentation.
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy

import multiprocessing as mp
import zmq
import ctypes
import numpy as np
import multiprocessing as mp
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

N = 8


def create_sample_device_data():
  a = numpy.random.randn(1, 32)
  a = a.astype(numpy.int8)
  print('orig: ', a)
  print('orig: ', a.shape)

  a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
  cuda.memcpy_htod(a_gpu, a)
  return a_gpu


def func1():
  drv.init()
  dev = drv.Device(0)
  ctx_gpu = dev.make_context()

  ctx = zmq.Context()
  sock = ctx.socket(zmq.REQ)
  sock.connect('tcp://localhost:6000')

  x_gpu = create_sample_device_data()
  h = drv.mem_get_ipc_handle(x_gpu)
  sock.send_pyobj(h)

  ctx_gpu.pop()


def func2():
  drv.init()
  dev = drv.Device(0)
  ctx_gpu = dev.make_context()

  ctx = zmq.Context()
  sock = ctx.socket(zmq.REP)
  sock.bind('tcp://*:6000')

  h = sock.recv_pyobj()

  x_ptr = drv.IPCMemoryHandle(h)
  x_gpu = gpuarray.GPUArray((1, 32), numpy.int8, gpudata=x_ptr)

  print('gpu:  ', x_gpu.get())

  ctx_gpu.pop()


if __name__ == '__main__':
  p1 = mp.Process(target=func1)
  p2 = mp.Process(target=func2)

  p1.start()
  p2.start()

