import flatbuffers

from blazingdb.protocol.errors import Error
import blazingdb.protocol.transport
import blazingdb.protocol.transport as transport
from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.messages.blazingdb.protocol.ResponseError import ResponseError
from blazingdb.messages.blazingdb.protocol \
    import TableGroup, BlazingTable

from blazingdb.messages.blazingdb.protocol.orchestrator \
    import DMLRequest, DMLResponse, DDLRequest, DDLResponse, DDLCreateTableRequest, DDLDropTableRequest
from blazingdb.messages.blazingdb.protocol.orchestrator.MessageType \
  import MessageType as OrchestratorMessageType

from blazingdb.messages.blazingdb.protocol.orchestrator \
  import AuthRequest, AuthResponse

from blazingdb.messages.blazingdb.protocol.gdf \
  import gdf_column_handler

class gdf_column_handlerSchema(transport.schema(gdf_column_handler)):
  size = transport.NumberSegment()

class BlazingTableSchema(transport.schema(BlazingTable)):
  name: transport.StringSegment()
  columns = transport.VectorSchemaSegment(gdf_column_handlerSchema)
  columnNames: transport.VectorStringSegment(transport.StringSegment)

class TableGroupSchema(transport.schema(TableGroup)):
  tables = transport.VectorSchemaSegment(BlazingTableSchema)
  name = transport.StringSegment()

class DMLRequestSchema(transport.schema(DMLRequest)):
  query = transport.StringSegment()
  groupTable = transport.SchemaSegment(TableGroupSchema)

class DDLRequestSchema(transport.schema(DDLRequest)):
  query = transport.StringSegment()

class DDLCreateTableRequestSchema(transport.schema(DDLCreateTableRequest)):
  name = transport.StringSegment()
  columnNames = transport.VectorStringSegment(transport.StringSegment)
  columnTypes = transport.VectorStringSegment(transport.StringSegment)
  dbName = transport.StringSegment()

class DDLDropTableRequestSchema(transport.schema(DDLDropTableRequest)):
  name = transport.StringSegment()
  dbName = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.NumberSegment()

class AuthResponseSchema(transport.schema(AuthResponse)):
  accessToken = transport.NumberSegment()

class AuthRequestSchema(transport.schema(AuthRequest)):
  pass
