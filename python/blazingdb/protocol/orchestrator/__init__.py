import flatbuffers

from blazingdb.messages.blazingdb.protocol import Request, Response, Status
from blazingdb.messages.blazingdb.protocol.orchestrator \
  import (MessageType, DMLRequest)


class DMLRequestDTO:

  def __init__(self, query):
    self.query = query


class RequestDTO:

  def __init__(self, header, payload):
    self.header = header
    self.payload = payload


class ResponseDTO:

  def __init__(self, status, payload):
    self.status = status
    self.payload = payload


def MakeDMLRequest(query):
  builder = flatbuffers.Builder(1024)
  query = builder.CreateString(query)
  DMLRequest.DMLRequestStart(builder)
  DMLRequest.DMLRequestAddQuery(builder, query)
  builder.Finish(DMLRequest.DMLRequestEnd(builder))
  buffer_ = builder.Output()

  builder = flatbuffers.Builder(32)

  Request.RequestStartPayloadVector(builder, len(buffer_))
  for b in reversed(buffer_):
    builder.PrependByte(b)
  payload = builder.EndVector(len(buffer_))

  Request.RequestStart(builder)
  Request.RequestAddHeader(builder, MessageType.MessageType.DML)
  Request.RequestAddPayload(builder, payload)
  builder.Finish(Request.RequestEnd(builder))

  return builder.Output()


def DMLRequestFrom(buffer_):
  request = Request.Request.GetRootAsRequest(buffer_, 0)
  payload = []
  for i in range(request.PayloadLength()):
    payload.append(request.Payload(i))
  payload = bytes(payload)

  dml = DMLRequest.DMLRequest.GetRootAsDMLRequest(payload, 0)

  return RequestDTO(request.Header(), DMLRequestDTO(dml.Query()))


def MakeDMLResponse():
  builder = flatbuffers.Builder(16)
  Response.ResponseStart(builder)
  Response.ResponseAddStatus(builder, Status.Status.Success)
  builder.Finish(Response.ResponseEnd(builder))
  return builder.Output()


def DMLResponseFrom(buffer_):
  response = Response.Response.GetRootAsResponse(buffer_, 0)
  return ResponseDTO(response.Status(), None)
