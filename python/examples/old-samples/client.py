import blazingdb.protocol
import blazingdb.protocol.interpreter
import blazingdb.protocol.orchestrator
import blazingdb.protocol.transport.channel

from blazingdb.protocol.interpreter import InterpreterMessage

ACCESS_TOKEN = 456


def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/socket')
  client = blazingdb.protocol.Client(connection)

  getResult = blazingdb.protocol.interpreter.GetResultRequestSchema(
    resultToken=123456)
  requestBuffer = blazingdb.protocol.transport.channel.MakeRequestBuffer(
    InterpreterMessage.GetResult, ACCESS_TOKEN, getResult)

  responseBuffer = client.send(requestBuffer)

  response = blazingdb.protocol.transport.channel.ResponseSchema.From(
    responseBuffer)

  print(response.status)

  dmlResponse = blazingdb.protocol.orchestrator.DMLResponseSchema.From(
    response.payload)

  print(dmlResponse.resultToken)

  responseBuffer = client.send(requestBuffer)

if __name__ == '__main__':
  main()