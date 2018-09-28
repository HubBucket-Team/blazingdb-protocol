import flatbuffers

from blazingdb.protocol.messages import DMLRequest


def MakeDMLRequest(query):
  builder = flatbuffers.Builder(0)
  query = builder.CreateString(query)
  DMLRequest.DMLRequestStart(builder)
  DMLRequest.DMLRequestAddQuery(builder, query)
  builder.Finish(DMLRequest.DMLRequestEnd(builder))
  return builder.Output()

def DMLRequestFrom(_buffer):
  dmlRequest = DMLRequest.DMLRequest.GetRootAsDMLRequest(_buffer, 0)
  print(dmlRequest.Query())
