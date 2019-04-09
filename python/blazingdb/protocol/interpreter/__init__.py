import flatbuffers
import copy
import blazingdb.protocol.transport as transport
import numpy
from blazingdb.messages.blazingdb.protocol.interpreter \
  import (DMLRequest, DMLResponse, GetResultRequest, GetResultResponse,
           BlazingMetadata)

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage

from blazingdb.protocol.gdf import gdf_columnSchema

class DMLRequestSchema(transport.schema(DMLRequest)):
  logicalPlan = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.StringSegment()


class GetResultRequestSchema(transport.schema(GetResultRequest)):
  resultToken = transport.NumberSegment()

class BlazingMetadataSchema(transport.schema(BlazingMetadata)):
  status = transport.StringSegment()
  message = transport.StringSegment()
  time = transport.NumberSegment()
  rows = transport.NumberSegment()


class GetResultResponseSchema(transport.schema(GetResultResponse)):
  metadata = transport.SchemaSegment(BlazingMetadataSchema)
  columns = transport.VectorGdfColumnSegment(gdf_columnSchema)
  columnNames = transport.VectorStringSegment(transport.StringSegment)
  columnTokens = transport.VectorSegment(transport.NumberSegment)

def _get_bytearray(ipch):
  nr_of_bytes = ipch.ReservedLength()
  np_buffer = numpy.empty([nr_of_bytes, ], dtype=numpy.uint8)
  for i in range(nr_of_bytes):
    np_buffer[i] = ipch.Reserved(i)
  return bytearray(np_buffer)

def _get_dtype_info(dtype_info):
  response = dtype_info
  response['custrings_views'] = _get_bytearray(dtype_info.custrings_views)
  response['custrings_membuffer'] = _get_bytearray(dtype_info.custrings_membuffer)
  return response

def GetQueryResultFrom(payloadBuffer):
  result = GetResultResponseSchema.From(payloadBuffer)
  columns = []
  column_list = list(item for item in result.columns)
  for item in column_list:
    column = copy.deepcopy(item)
    column.data = _get_bytearray(item.data)
    column.valid = _get_bytearray(item.valid)
    column.dtype_info = _get_dtype_info(item.dtype_info)
    columns.append(column)
  return type('obj', (object,), {
    'metadata': result.metadata,
    'columnNames': list(result.columnNames),
    'columnTokens': list(result.columnTokens),
    'columns': columns
  })