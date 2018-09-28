import flatbuffers

import blazingdb.protocol.transport

from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.messages.blazingdb.protocol.orchestrator \
    import DMLRequest, DMLResponse

from blazingdb.messages.blazingdb.protocol.orchestrator.MessageType \
  import MessageType as OrchestratorMessageType


class DMLRequestDTO:

  def __init__(self, query):
    self.query = query


class DMLResponseDTO:

  def __init__(self, token):
    self.token = token


def MakeDMLRequest(query):
  builder = flatbuffers.Builder(512)
  query = builder.CreateString(query)
  DMLRequest.DMLRequestStart(builder)
  DMLRequest.DMLRequestAddQuery(builder, query)
  builder.Finish(DMLRequest.DMLRequestEnd(builder))
  return blazingdb.protocol.transport.MakeRequest(
    blazingdb.protocol.transport.RequestDTO(OrchestratorMessageType.DML,
                                            builder.Output()), 512)


def DMLRequestFrom(buffer_):
  request = blazingdb.protocol.transport.RequestFrom(buffer_)
  dmlRequest = DMLRequest.DMLRequest.GetRootAsDMLRequest(request.payload, 0)
  return blazingdb.protocol.transport.RequestDTO(
    request.header, DMLRequestDTO(dmlRequest.Query()))


def MakeDMLResponse(token):
  builder = flatbuffers.Builder(512)
  token = builder.CreateString(token)
  DMLResponse.DMLResponseStart(builder)
  DMLResponse.DMLResponseAddToken(builder, token)
  builder.Finish(DMLResponse.DMLResponseEnd(builder))
  return blazingdb.protocol.transport.MakeResponse(
    blazingdb.protocol.transport.ResponseDTO(Status.Success,
                                             builder.Output()), 512)


def DMLResponseFrom(buffer_):
  response = blazingdb.protocol.transport.ResponseFrom(buffer_)
  dmlResponse = DMLResponse.DMLResponse.GetRootAsDMLResponse(response.payload, 0)
  return blazingdb.protocol.transport.ResponseDTO(
      response.status, DMLResponseDTO(dmlResponse.Token()))
