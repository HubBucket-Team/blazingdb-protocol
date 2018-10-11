import flatbuffers

import blazingdb.protocol.transport as transport
import blazingdb.protocol.transport

from blazingdb.messages.blazingdb.protocol.interpreter \
  import (DMLRequest, DMLResponse, GetResultRequest, GetResultResponse,
           BlazingMetadata)

from blazingdb.messages.blazingdb.protocol.gdf \
  import gdf_column_handler

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage


class DMLRequestSchema(transport.schema(DMLRequest)):
  logicalPlan = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.StringSegment()


class GetResultRequestSchema(transport.schema(GetResultRequest)):
  resultToken = transport.NumberSegment()


class gdf_column_handlerSchema(transport.schema(gdf_column_handler)):
  size = transport.NumberSegment()


class BlazingMetadataSchema(transport.schema(BlazingMetadata)):
  status = transport.StringSegment()
  message = transport.StringSegment()
  time = transport.NumberSegment()
  rows = transport.NumberSegment()


class GetResultResponseSchema(transport.schema(GetResultResponse)):
  metadata = transport.SchemaSegment(BlazingMetadataSchema)
  fieldNames = transport.VectorSegment(transport.StringSegment)
  values = transport.VectorSchemaSegment(gdf_column_handlerSchema)
