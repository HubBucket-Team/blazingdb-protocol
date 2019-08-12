import flatbuffers

from blazingdb.protocol.errors import Error
import blazingdb.protocol.transport
import blazingdb.protocol.transport as transport
from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.messages.blazingdb.protocol.ResponseError import ResponseError
from blazingdb.messages.blazingdb.protocol \
  import TableGroup, BlazingTable

from blazingdb.messages.blazingdb.protocol.orchestrator \
  import DMLRequest, DMLResponse, DDLResponse, DDLCreateTableRequest, DDLDropTableRequest, DMLDistributedResponse

from blazingdb.messages.blazingdb.protocol.orchestrator.MessageType \
  import MessageType as OrchestratorMessageType

from blazingdb.messages.blazingdb.protocol \
  import NodeConnection

from blazingdb.messages.blazingdb.protocol.orchestrator \
  import AuthRequest, AuthResponse

from blazingdb.protocol.gdf import gdf_columnSchema


class BlazingTableSchema(transport.schema(BlazingTable)):
  columns = transport.VectorSchemaSegment(gdf_columnSchema)
  columnTokens = transport.VectorSegment(transport.NumberSegment)
  resultToken = transport.NumberSegment()

class TableGroupSchema(transport.schema(TableGroup)):
  tables = transport.VectorSchemaSegment(BlazingTableSchema)
  name = transport.StringSegment()

class DMLRequestSchema(transport.schema(DMLRequest)):
  query = transport.StringSegment()
  tableGroup = transport.SchemaSegment(TableGroupSchema)

class DDLCreateTableRequestSchema(transport.schema(DDLCreateTableRequest)):
  name = transport.StringSegment()
  columnNames = transport.VectorStringSegment(transport.StringSegment)
  columnTypes = transport.VectorStringSegment(transport.StringSegment)
  dbName = transport.StringSegment()
  schemaType = transport.NumberSegment()
  gdf = transport.SchemaSegment(BlazingTableSchema)
  files = transport.VectorStringSegment(transport.StringSegment)
  csvDelimiter = transport.StringSegment()
  csvLineTerminator = transport.StringSegment()
  csvSkipRows = transport.NumberSegment()
  resultToken = transport.NumberSegment()
  csvHeader = transport.NumberSegment()

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


class DMLDistributedResponseSchema(transport.schema(DMLDistributedResponse)):
    responses = transport.VectorSchemaSegment(DMLResponseSchema)


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

      if 'custrings_data' not in c:
        custrings_data = blazingdb.protocol.gdf.custringsData_tSchema(reserved=b'')
      else:
        custrings_data = blazingdb.protocol.gdf.custringsData_tSchema(reserved=c['custrings_data'])

      dtype_info = blazingdb.protocol.gdf.gdf_dtype_extra_infoSchema(time_unit=0)
      gdfColumn = blazingdb.protocol.gdf.gdf_columnSchema(data=data, valid=valid,
                                size=c['size'],
                                dtype=c['dtype'], dtype_info=dtype_info,
                                null_count=c['null_count'],
                                custrings_data=custrings_data)
      columns.append(gdfColumn)
    table = blazingdb.protocol.orchestrator.BlazingTableSchema(name=tableName, columns=columns,
                                 columnNames=columnNames, columnTokens=columnTokens, resultToken=resultToken)
    tables.append(table)
  tableGroup = blazingdb.protocol.orchestrator.TableGroupSchema(tables=tables, name=tableGroupName)
  return blazingdb.protocol.orchestrator.DMLRequestSchema(query=query, tableGroup=tableGroup)

def BuildDDLCreateTableRequestSchema(name, columnNames, columnTypes, dbName, schemaType, gdf, files, csvDelimiter, csvLineTerminator, csvSkipRows,resultToken,
  csvHeader):
  if(resultToken == 0):
    resultToken = gdf['resultToken']	
  columnTokens = gdf['columnTokens']
  columns = []
  for i, c in enumerate(gdf['columns']):
    if c['data'] is None:
      data = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=b'')
    else:
      data = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=c['data'])
    if c['valid'] is None:
      valid = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=b'')
    else:
      valid = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved=c['valid'])

    if 'custrings_data' not in c or c['custrings_data'] is None:
      custrings_data = blazingdb.protocol.gdf.custringsData_tSchema(reserved=b'')
    else:
      custrings_data = blazingdb.protocol.gdf.custringsData_tSchema(reserved=c['custrings_data'])

    dtype_info = blazingdb.protocol.gdf.gdf_dtype_extra_infoSchema(time_unit=0)
    gdfColumn = blazingdb.protocol.gdf.gdf_columnSchema(data=data, valid=valid,
                              size=c['size'],
                              dtype=c['dtype'], dtype_info=dtype_info,
                              null_count=c['null_count'],
                              custrings_data=custrings_data)
    columns.append(gdfColumn)
  table = blazingdb.protocol.orchestrator.BlazingTableSchema(columns=columns, columnTokens=columnTokens, resultToken=resultToken)

  return blazingdb.protocol.orchestrator.DDLCreateTableRequestSchema(name=name,
                                                                                       columnNames=columnNames,
                                                                                       columnTypes=columnTypes,
                                                                                       dbName=dbName,
                                                                                       schemaType=schemaType,
                                                                                       gdf=table,
                                                                                       files=files,
                                                                                       csvDelimiter=csvDelimiter,
                                                                                       csvLineTerminator=csvLineTerminator,
                                                                                       csvSkipRows=csvSkipRows,
                                                                                       resultToken=resultToken,
                                                                                       csvHeader=csvHeader,)
