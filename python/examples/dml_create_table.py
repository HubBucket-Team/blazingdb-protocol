import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel
from blazingdb.protocol.errors import Error
from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.orchestrator import OrchestratorMessageType


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

    print(bytes(requestBuffer).hex())
    responseBuffer = self._send_request(self._orchestrator_path, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    print(response.status)
    return response.status

  def run_ddl_drop_table(self, query):
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

def dml_create_table_example(tableName, columnNames, columnTypes, dbName):
  dmlRequestSchema = blazingdb.protocol.orchestrator.DDLCreateTableRequestSchema(name=tableName,
                                                                                 columnNames=columnNames,
                                                                                 columnTypes=columnTypes, dbName=dbName)
  payload = dmlRequestSchema.ToBuffer()
  response = blazingdb.protocol.orchestrator.DDLCreateTableRequestSchema.From(payload)

  print(response.name)
  print(response.dbName)
  print(list(response.columnNames))
  print(list(response.columnTypes))

def main():
  dml_create_table_example('user', ['name', 'surname', 'age'], ['string', 'string', 'int'], 'alexdb')

  client = PyConnector('/tmp/orchestrator.socket', '/tmp/ral.socket')

  try:
    client.connect()
  except Error as err:
    print(err)

  try:
    client.run_ddl_create_table('user', ['name', 'surname', 'age'], ['string', 'string', 'int'], 'alexdb')
  except Error as err:
    print(err)

  # try:
  #   client.run_ddl_drop_table('user')
  # except Error as err:
  #   print(err)

  client.close_connection()

if __name__ == '__main__':
  main()
