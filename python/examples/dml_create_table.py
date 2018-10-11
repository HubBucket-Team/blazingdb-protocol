import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel
from blazingdb.protocol.errors import Error
from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.orchestrator import OrchestratorMessageType

from blazingdb.protocol.gdf import gdf_columnSchema

class PyConnector:
  def __init__(self, orchestrator_path, interpreter_path):
    self._orchestrator_path = orchestrator_path
    self._interpreter_path = interpreter_path

  def connect(self):
    print("open connection")
    authSchema = blazingdb.protocol.orchestrator.AuthRequestSchema()

    requestBuffer = blazingdb.protocol.transport.channel.MakeAuthRequestBuffer(
      OrchestratorMessageType.AuthOpen, authSchema)

    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)

    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      print(errorResponse.errors)
      raise Error(errorResponse.errors)
    responsePayload = blazingdb.protocol.orchestrator.AuthResponseSchema.From(response.payload)
    print(responsePayload.accessToken)
    self.accessToken = responsePayload.accessToken

  def _send_request(self, unix_path, requestBuffer):
    connection = blazingdb.protocol.UnixSocketConnection(unix_path)
    client = blazingdb.protocol.Client(connection)
    return client.send(requestBuffer)

  def run_dml_query(self, query):
    print(query)
    dmlRequestSchema = blazingdb.protocol.orchestrator.DMLRequestSchema(query=query)
    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(OrchestratorMessageType.DML,
                                                                           self.accessToken, dmlRequestSchema)
    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    dmlResponseDTO = blazingdb.protocol.orchestrator.DMLResponseSchema.From(response.payload)
    print(dmlResponseDTO.resultToken)
    self._get_result(dmlResponseDTO.resultToken)

  def run_ddl_query(self, query):
    print(query)
    dmlRequestSchema = blazingdb.protocol.orchestrator.DDLRequestSchema(query=query)
    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(OrchestratorMessageType.DDL,
                                                                           self.accessToken, dmlRequestSchema)
    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    print(response.status)
    return response.status

  def run_ddl_create_table(self, tableName, columnNames, columnTypes, dbName):
    print(tableName)
    dmlRequestSchema = blazingdb.protocol.orchestrator.DDLCreateTableRequestSchema(name=tableName, columnNames=columnNames, columnTypes=columnTypes, dbName=dbName)
    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(OrchestratorMessageType.DDL_CREATE_TABLE,
                                                                           self.accessToken, dmlRequestSchema)
    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    print(response.status)
    return response.status

  def run_ddl_drop_table(self, tableName, dbName):
    print(tableName)
    dmlRequestSchema = blazingdb.protocol.orchestrator.DDLDropTableRequestSchema(name=tableName, dbName=dbName)
    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(OrchestratorMessageType.DDL_DROP_TABLE,
                                                                           self.accessToken, dmlRequestSchema)
    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    print(response.status)
    return response.status

  def close_connection(self):
    print("close connection")
    authSchema = blazingdb.protocol.orchestrator.AuthRequestSchema()

    requestBuffer = blazingdb.protocol.transport.channel.MakeAuthRequestBuffer(
      OrchestratorMessageType.AuthClose, authSchema)

    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      print(errorResponse.errors)
    print(response.status)

  def _get_result(self, result_token):

    getResultRequest = blazingdb.protocol.interpreter.GetResultRequestSchema(
      resultToken=result_token)

    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(
      InterpreterMessage.GetResult, self.accessToken, getResultRequest)

    responseBuffer = self._send_request(self._interpreter_path, requestBuffer)

    response = blazingdb.protocol.transport.channel.ResponseSchema.From(
      responseBuffer)

    if response.status == Status.Error:
      raise ValueError('Error status')

    getResultResponse = \
      blazingdb.protocol.interpreter.GetResultResponseSchema.From(
        response.payload)

    print('GetResult Response')
    print('  metadata:')
    print('     status: %s' % getResultResponse.metadata.status)
    print('    message: %s' % getResultResponse.metadata.message)
    print('       time: %s' % getResultResponse.metadata.time)
    print('       rows: %s' % getResultResponse.metadata.rows)
    print('  fieldNames: %s' % list(getResultResponse.fieldNames))
    print('  values:')
    print('    size: %s' % [value.size for value in getResultResponse.values])

def ddl_create_table_example(tableName, columnNames, columnTypes, dbName):
  dmlRequestSchema = blazingdb.protocol.orchestrator.DDLCreateTableRequestSchema(name=tableName,
                                                                                 columnNames=columnNames,
                                                                                 columnTypes=columnTypes, dbName=dbName)
  payload = dmlRequestSchema.ToBuffer()
  response = blazingdb.protocol.orchestrator.DDLCreateTableRequestSchema.From(payload)

  print(response.name)
  print(response.dbName)
  print(list(response.columnNames))
  print(list(response.columnTypes))


from blazingdb.protocol.transport import schema, NumberSegment, SchemaSegment
from blazingdb.messages.blazingdb.protocol.interpreter import BlazingMetadata, GetResultResponse

def create_query():
  query = 'select id, age from $0'
  db = {
    'name': 'alexdb',
    'tables': [
      {
        'name': 'user',
        'columns': [{'data': 0, 'valid': 0, 'size': 0, 'dtype': 0, 'dtype_info': 0},
                    {'data': 0, 'valid': 0, 'size': 20, 'dtype': 1, 'dtype_info': 1}],
        'columnNames': ['id', 'age']
      },
      {
        'name': 'computer',
        'columns': [{'data': 0, 'valid': 0, 'size': 0, 'dtype': 0, 'dtype_info': 0},
                    {'data': 0, 'valid': 0, 'size': 20, 'dtype': 1, 'dtype_info': 1}],
        'columnNames': ['id', 'brand']
      }
    ]
  }
  print(query)
  print(db['tables'][0]['columnNames'])

  data = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved='data'.encode())
  valid = blazingdb.protocol.gdf.cudaIpcMemHandle_tSchema(reserved='valid'.encode())
  dtype_info = blazingdb.protocol.gdf.gdf_dtype_extra_infoSchema(time_unit=0)
  gdfColumn = blazingdb.protocol.gdf.gdf_columnSchema(data=data, valid=valid, size=10, dtype=0, dtype_info=dtype_info, null_count = 0)

  cloneGdf = blazingdb.protocol.gdf.gdf_columnSchema.From(gdfColumn.ToBuffer())
  print(cloneGdf.size)

  table1 = blazingdb.protocol.orchestrator.BlazingTableSchema(name=db['name'], columns=[gdfColumn, gdfColumn], columnNames=['id', 'age'])

  cloneTable1 = blazingdb.protocol.orchestrator.BlazingTableSchema.From(table1.ToBuffer())
  print(cloneTable1.name)

  # table2 = blazingdb.protocol.orchestrator.BlazingTableSchema(name=db['name'], columns=[gdfColumn, gdfColumn], columnNames=db['tables'][0]['columnNames'])
  tableGroup = blazingdb.protocol.orchestrator.TableGroupSchema(tables=[table1], name='alexdb')
  dmlRequest = blazingdb.protocol.orchestrator.DMLRequestSchema(query=query, tableGroup=tableGroup)
  payload = dmlRequest.ToBuffer()
  response = blazingdb.protocol.orchestrator.DMLRequestSchema.From(payload)

  print(response.query)
  print(response.tableGroup)
  print(list(response.tableGroup.name))

  class B(schema(BlazingMetadata)):
    rows = NumberSegment()

  class A(schema(GetResultResponse)):
    metadata = SchemaSegment(B)

  b = B(rows=1235)
  a = A(metadata=b)

  ll = a.ToBuffer()

  print(A.From(ll).metadata.rows)

def main():
  ddl_create_table_example('user', ['name', 'surname', 'age'], ['string', 'string', 'int'], 'alexdb')

  create_query()

  # client = PyConnector('/tmp/orchestrator.socket', '/tmp/ral.socket')
  #
  # try:
  #   client.connect()
  # except Error as err:
  #   print(err)
  #
  # try:
  #   client.run_dml_query('select * from Table')
  # except SyntaxError as err:
  #   print(err)
  #
  # try:
  #   client.run_ddl_create_table('user', ['name', 'surname', 'age'], ['string', 'string', 'int'], 'alexdb')
  # except Error as err:
  #   print(err)
  #
  # try:
  #   client.run_ddl_drop_table('user', 'alexdb')
  # except Error as err:
  #   print(err)
  #
  # client.close_connection()

if __name__ == '__main__':
  main()
