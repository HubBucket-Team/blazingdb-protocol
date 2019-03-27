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

from blazingdb.messages.blazingdb.protocol \
  import NodeConnection

from blazingdb.messages.blazingdb.protocol.orchestrator \
  import AuthRequest, AuthResponse

from blazingdb.protocol.gdf import gdf_columnSchema


class BlazingTableSchema(transport.schema(BlazingTable)):
  name = transport.StringSegment()
  columns = transport.VectorSchemaSegment(gdf_columnSchema)
  columnNames = transport.VectorStringSegment(transport.StringSegment)
  columnTokens = transport.VectorSegment(transport.NumberSegment)
  resultToken = transport.NumberSegment()

class TableGroupSchema(transport.schema(TableGroup)):
  tables = transport.VectorSchemaSegment(BlazingTableSchema)
  name = transport.StringSegment()

class DMLRequestSchema(transport.schema(DMLRequest)):
  query = transport.StringSegment()
  tableGroup = transport.SchemaSegment(TableGroupSchema)

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

class NodeConnectionSchema(transport.schema(NodeConnection)):
  path = transport.StringSegment()
  port = transport.NumberSegment()
  type = transport.NumberSegment()

class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.NumberSegment()
  nodeConnection = transport.SchemaSegment(NodeConnectionSchema)
  calciteTime = transport.NumberSegment()

class AuthResponseSchema(transport.schema(AuthResponse)):
  accessToken = transport.NumberSegment()

class AuthRequestSchema(transport.schema(AuthRequest)):
  pass


def BuildDMLRequestSchema(query, tableGroupDto):
  tableGroupName = tableGroupDto['name']
  tables = []
  for index, t in enumerate(tableGroupDto['tables']):
    tableName = t['name']
    resultToken = t['resultToken']
    columnNames = t['columnNames']
    columnTokens = t['columnTokens']
    columns = []
    for i, c in enumerate(t['columns']):
      if c['data'] is None:
        data = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=b'')
      else:
        data = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=c['data'])
      if c['valid'] is None:
        valid = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=b'')
      else:
        valid = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=c['valid'])
      if c['custrings_membuffer'] is None:
        custrings_membuffer = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=b'')
      else:
        custrings_membuffer = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=c['custrings_membuffer'])
      if c['custrings_views'] is None:
        custrings_views = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=b'')
      else:
        custrings_views = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=c['custrings_views'])

      dtype_info = blazingdb.protocol.gdf.gdf_dtype_extra_infoSchema(time_unit=c['dtype_info'])
      gdfColumn = blazingdb.protocol.gdf.gdf_columnSchema(data=data, valid=valid,
                                custrings_membuffer=custrings_membuffer,
                                custrings_views=custrings_views, size=c['size'],
                                dtype=c['dtype'], dtype_info=dtype_info,
                                null_count=c['null_count'])
      columns.append(gdfColumn)
    table = blazingdb.protocol.orchestrator.BlazingTableSchema(name=tableName, columns=columns,
                                 columnNames=columnNames, columnTokens=columnTokens, resultToken=resultToken)
    tables.append(table)
  tableGroup = blazingdb.protocol.orchestrator.TableGroupSchema(tables=tables, name=tableGroupName)
  return blazingdb.protocol.orchestrator.DMLRequestSchema(query=query, tableGroup=tableGroup)
