import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.authorization
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel
from blazingdb.protocol.errors import Error

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.orchestrator import OrchestratorMessageType


class PyConnector:
  def __init__ (self, path):
    self.unixPath = path
    self.client = self._open_client(self.unixPath)
    self.accessToken = 0
    self._connect()

  def _connect(self):

    authSchema = blazingdb.protocol.orchestrator.AuthRequestSchema()

    requestBuffer = blazingdb.protocol.transport.channel.MakeAuthRequestBuffer(
      OrchestratorMessageType.AuthOpen, authSchema)

    try:
      responseBuffer = self.client.send(requestBuffer)
      response = blazingdb.protocol.orchestrator.AuthResponseFrom(responseBuffer)
      print(response.payload.accessToken)
      self.accessToken = response.payload.accessToken
    except Error as err:
      print(err)

  def _open_client(self, unixPath) :
    connection = blazingdb.protocol.UnixSocketConnection(unixPath)
    return blazingdb.protocol.Client(connection)

  def run_dml_query(self, query):
    self.client = self._open_client(self.unixPath)
    requestBuffer = blazingdb.protocol.orchestrator.MakeDMLRequest(self.accessToken, query)
    responseBuffer = self.client.send(requestBuffer)
    try:
      response = blazingdb.protocol.orchestrator.DMLResponseFrom(responseBuffer)
      print(response.payload.token)
      return response.payload.token
    except Error as err:
      print(err)

  def run_ddl_query(self, query):
    self.client = self._open_client(self.unixPath)
    requestBuffer = blazingdb.protocol.orchestrator.MakeDDLRequest(self.accessToken, query)
    responseBuffer = self.client.send(requestBuffer)
    try:
      response = blazingdb.protocol.orchestrator.DDLResponseFrom(responseBuffer)
      print(response.status)
      return response.status
    except Error as err:
      print(err)
    return ''

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
  connector.run_dml_query('select * from Table')
  connector.run_dml_query('@typo * from Table')

  connector.run_ddl_query('create database alexdb')
  connector.run_ddl_query('@typo database alexdb')


if __name__ == '__main__':
  main()
