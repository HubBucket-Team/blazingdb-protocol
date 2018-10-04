import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.authorization
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel
from blazingdb.protocol.errors import Error
from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.orchestrator import OrchestratorMessageType


class PyConnector:
  def __init__ (self, path):
    self.unixPath = path
    self.accessToken = 0
    self._connect()

  def send_request(self, unixPath, requestBuffer):
    connection = blazingdb.protocol.UnixSocketConnection(unixPath)
    client = blazingdb.protocol.Client(connection)
    return client.send(requestBuffer)

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

  def _get_result(self, result_token):
    self.client = self._open_client(self.unixPath)

    getResult = blazingdb.protocol.interpreter.GetResultSchema(
      token=result_token)

    requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(
      InterpreterMessage.GetResult, self.accessToken, getResult)

    responseBuffer = self.client.send(requestBuffer)

    return blazingdb.protocol.orchestrator.DMLResponseFrom(responseBuffer)


def main():
  connector = PyConnector('/tmp/orchestrator.socket')
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

if __name__ == '__main__':
  main()
