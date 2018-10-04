import flatbuffers

import blazingdb.protocol.transport as transport
import blazingdb.protocol.transport

from blazingdb.messages.blazingdb.protocol.interpreter \
  import (DMLRequest, DMLResponse, GetResultRequest, GetResultResponse,
          gdf_column, BlazingMetadata)

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage


class DMLRequestSchema(transport.schema(DMLRequest)):
  logicalPlan = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.StringSegment()


class GetResultRequestSchema(transport.schema(GetResultRequest)):
  resultToken = transport.StringSegment()


class gdf_columnSchema(transport.schema(gdf_column)):
  size = transport.NumberSegment()


class BlazingMetadataSchema(transport.schema(BlazingMetadata)):
  status = transport.StringSegment()
  message = transport.StringSegment()
  time = transport.NumberSegment()
  rows = transport.NumberSegment()


class GetResultResponseSchema(transport.schema(GetResultResponse)):
  metadata = transport.SchemaSegment(BlazingMetadataSchema)
  fieldNames = transport.VectorSegment(transport.StringSegment)
  values = transport.VectorSchemaSegment(gdf_columnSchema)
