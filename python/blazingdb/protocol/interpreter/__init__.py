import flatbuffers

import blazingdb.protocol.transport as transport
import blazingdb.protocol.transport

from blazingdb.messages.blazingdb.protocol.interpreter \
  import (DMLRequest, DMLResponse, GetResultRequest, GetResultResponse,
           BlazingMetadata)

from blazingdb.messages.blazingdb.protocol.interpreter.gdf \
  import gdf_column, cudaIpcMemHandle_t, gdf_dtype_extra_info

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage


class DMLRequestSchema(transport.schema(DMLRequest)):
  logicalPlan = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.StringSegment()


class GetResultRequestSchema(transport.schema(GetResultRequest)):
  resultToken = transport.NumberSegment()


class cudaIpcMemHandle_tSchema(transport.schema(cudaIpcMemHandle_t)):
  reserved = transport.BytesSegment()


class gdf_dtype_extra_infoSchema(transport.schema(gdf_dtype_extra_info)):
  time_unit = transport.NumberSegment()


class gdf_columnSchema(transport.schema(gdf_column)):
  data = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  valid = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  size = transport.NumberSegment()
  dtype = transport.NumberSegment()
  dtype_info = transport.SchemaSegment(gdf_dtype_extra_infoSchema)


class BlazingMetadataSchema(transport.schema(BlazingMetadata)):
  status = transport.StringSegment()
  message = transport.StringSegment()
  time = transport.NumberSegment()
  rows = transport.NumberSegment()


class GetResultResponseSchema(transport.schema(GetResultResponse)):
  metadata = transport.SchemaSegment(BlazingMetadataSchema)
  fieldNames = transport.VectorSegment(transport.StringSegment)
  values = transport.VectorSchemaSegment(gdf_columnSchema)
