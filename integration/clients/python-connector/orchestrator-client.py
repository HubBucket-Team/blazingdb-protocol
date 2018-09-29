import blazingdb.protocol
import blazingdb.protocol.orchestrator

def dml_request_example (client, query):
  requestBuffer = blazingdb.protocol.orchestrator.MakeDMLRequest(query)

  responseBuffer = client.send(requestBuffer)
  try :
    response = blazingdb.protocol.orchestrator.DMLResponseFrom(responseBuffer)
    print(response.payload.token)
  except ValueError as err:   
    print(err)

# def ddl_request_example (client, query):
#   requestBuffer = blazingdb.protocol.orchestrator.MakeDDLRequest(query)

#   responseBuffer = client.send(requestBuffer)
#   try :
#     response = blazingdb.protocol.orchestrator.DDLResponseFrom(responseBuffer)
#     print(response.payload.token)
#   except ValueError as err:   
#     print(err)
    
def main():
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/orchestrator.socket')
  client = blazingdb.protocol.Client(connection)

  query = 'select * from Table'
  dml_request_example(client, query)

  query = '@typo * from Table'
  dml_request_example(client, query)


if __name__ == '__main__':
  main()
