import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.authorization
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel

from blazingdb.protocol.interpreter import InterpreterMessage
from blazingdb.protocol.authorization import AuthorizationMessage


class PyConnector:
  def __init__ (self, path):
    self.unixPath = path
    self.client = self._open_client(self.unixPath)
    self.accessToken = 0
    self._connect()

  def _connect(self):

    authSchema = blazingdb.protocol.authorization.AuthRequestSchema()

    requestBuffer = blazingdb.protocol.transport.channel.MakeAuthRequestBuffer(
      AuthorizationMessage.Auth, authSchema)

    responseBuffer = self.client.send(requestBuffer)

    response = blazingdb.protocol.authorization.AuthResponseFrom(responseBuffer)
    print(response.payload.accessToken)
    self.accessToken = response.payload.accessToken

  def _open_client(self, unixPath) :
    connection = blazingdb.protocol.UnixSocketConnection(unixPath)
    return blazingdb.protocol.Client(connection)

  def run_query(self, query):
    self.client = self._open_client(self.unixPath)
    requestBuffer = blazingdb.protocol.orchestrator.MakeDMLRequest(self.accessToken, query)
    responseBuffer = self.client.send(requestBuffer)
    try:
      response = blazingdb.protocol.orchestrator.DMLResponseFrom(responseBuffer)
      print(response.payload.token)
      return response.payload.token
    except ValueError as err:
      print(err)
    return ''
    # return self._get_result(response.payload.token)

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
  handlers = connector.run_query('select * from Table')
  print(handlers)

if __name__ == '__main__':
  main()
