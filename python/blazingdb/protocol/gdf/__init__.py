import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.gdf \
  import gdf_column_handler, cudaIpcMemHandle_t, custringsData_t, gdf_dtype_extra_info

class cudaIpcMemHandle_tSchema(transport.schema(cudaIpcMemHandle_t)):
  reserved = transport.BytesSegment()

class custringsData_tSchema(transport.schema(custringsData_t)):
  reserved = transport.BytesSegment()

class gdf_dtype_extra_infoSchema(transport.schema(gdf_dtype_extra_info)):
  time_unit = transport.NumberSegment()

class gdf_columnSchema(transport.schema(gdf_column_handler)):
  data = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  valid = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  size = transport.NumberSegment()
  dtype = transport.NumberSegment()
  dtype_info = transport.SchemaSegment(gdf_dtype_extra_infoSchema)
  null_count = transport.NumberSegment()
  custrings_data = transport.SchemaSegment(custringsData_tSchema)
