import blazingdb.protocol
import blazingdb.protocol.orchestrator

def dml_request_example (query):
  connection = blazingdb.protocol.UnixSocketConnection('/tmp/orchestrator.socket')
  client = blazingdb.protocol.Client(connection)

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

  query = 'select * from Table'
  dml_request_example(query)

  # @todo check for exception : ex. not valid sql statement
  # @todo error for consecutive requests (error in the python client) 
  query = '@typo * from Table'
  dml_request_example(query)

if __name__ == '__main__':
  main()
