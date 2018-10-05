import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel
from blazingdb.protocol.errors import Error
from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.orchestrator import OrchestratorMessageType

class PyConnector:
  def __init__ (self, path, interpreter_path):
    self.unixPath = path
    self._interpreter_path = interpreter_path
    self._connect()

  def _connect(self):
    print("open connection")
    authSchema = blazingdb.protocol.orchestrator.AuthRequestSchema()

    requestBuffer = blazingdb.protocol.transport.channel.MakeAuthRequestBuffer(
      OrchestratorMessageType.AuthOpen, authSchema)

    responseBuffer = self.send_request(self.unixPath, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      print(errorResponse.errors)
    else:
      responsePayload = blazingdb.protocol.orchestrator.AuthResponseSchema.From(response.payload)
      print(responsePayload.accessToken)
      self.accessToken = responsePayload.accessToken

  def send_request(self, unixPath, requestBuffer):
    connection = blazingdb.protocol.UnixSocketConnection(unixPath)
    client = blazingdb.protocol.Client(connection)
    return client.send(requestBuffer)

  def run_dml_query(self, query):
    print(query)
    dmlRequestSchema = blazingdb.protocol.orchestrator.DMLRequestSchema(query = query)
    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(OrchestratorMessageType.DML, self.accessToken, dmlRequestSchema)
    responseBuffer = self.send_request(self.unixPath, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    dmlResponseDTO = blazingdb.protocol.orchestrator.DMLResponseSchema.From(response.payload)
    print(dmlResponseDTO.resultToken)
    return dmlResponseDTO.resultToken

  def run_ddl_query(self, query):
    print(query)
    dmlRequestSchema = blazingdb.protocol.orchestrator.DDLRequestSchema(query=query)
    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(OrchestratorMessageType.DDL,
                                                                           self.accessToken, dmlRequestSchema)
    responseBuffer = self.send_request(self.unixPath, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      raise Error(errorResponse.errors)
    print(response.status)
    return response.status

  def close_connection (self):
    print("close connection")
    authSchema = blazingdb.protocol.orchestrator.AuthRequestSchema()

    requestBuffer = blazingdb.protocol.transport.channel.MakeAuthRequestBuffer(
      OrchestratorMessageType.AuthClose, authSchema)

    responseBuffer = self.send_request(self.unixPath, requestBuffer)
    response = blazingdb.protocol.transport.channel.ResponseSchema.From(responseBuffer)
    if response.status == Status.Error:
      errorResponse = blazingdb.protocol.transport.channel.ResponseErrorSchema.From(response.payload)
      print(errorResponse.errors)
    print(response.status)

  def get_result(self, result_token):
    self.accessToken = 123

    getResultRequest = blazingdb.protocol.interpreter.GetResultRequestSchema(
      resultToken=result_token)

    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(
      InterpreterMessage.GetResult, self.accessToken, getResultRequest)

    responseBuffer = self.send_request(self._interpreter_path, requestBuffer)

    response = blazingdb.protocol.transport.channel.ResponseSchema.From(
      responseBuffer)

    if response.status == blazingdb.protocol.transport.channel.Status.Error:
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


def main():
  connector = PyConnector('/tmp/orchestrator.socket', '/tmp/ral.socket')
  try:
    connector.run_dml_query('select * from Table')
  except Error as err:
    print(err)

  try:
    connector.run_dml_query('@typo * from Table')
  except Error as err:
    print(err)

  try:
    connector.run_ddl_query('create database alexdb')
  except Error as err:
    print(err)
  try:
    connector.run_ddl_query('@typo database alexdb')
  except Error as err:
    print(err)

  connector.get_result('RESULT_TOKEN')

  connector.close_connection()

if __name__ == '__main__':
  main()
