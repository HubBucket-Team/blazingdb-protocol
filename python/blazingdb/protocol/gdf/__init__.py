import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.gdf \
  import gdf_column_handler, cudaIpcMemHandle_t, gdf_dtype_extra_info

class cudaIpcMemHandle_tSchema(transport.schema(cudaIpcMemHandle_t)):
  reserved = transport.BytesSegment()

class gdf_dtype_extra_infoSchema(transport.schema(gdf_dtype_extra_info)):
  time_unit = transport.NumberSegment()
  custrings_views = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  custrings_views_count = transport.NumberSegment()
  custrings_membuffer = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  custrings_membuffer_size = transport.NumberSegment()
  custrings_base_ptr = transport.NumberSegment()

class gdf_columnSchema(transport.schema(gdf_column_handler)):
  data = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  valid = transport.SchemaSegment(cudaIpcMemHandle_tSchema)
  size = transport.NumberSegment()
  dtype = transport.NumberSegment()
  dtype_info = transport.SchemaSegment(gdf_dtype_extra_infoSchema)
  null_count = transport.NumberSegment()

