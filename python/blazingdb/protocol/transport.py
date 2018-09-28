import flatbuffers

from blazingdb.messages.blazingdb.protocol import Request, Response


__all__ = ('RequestDTO', 'ResponseDTO', 'MakeRequest', 'RequestFrom')


class RequestDTO:

  def __init__(self, header, payload):
    self.header = header
    self.payload = payload


class ResponseDTO:

  def __init__(self, status, payload):
    self.status = status
    self.payload = payload


def MakeRequest(dto, builderInitialSize=0):
  builder = flatbuffers.Builder(builderInitialSize)
  header = dto.header
  payload = _CreatePayload(builder, dto.payload)
  Request.RequestStart(builder)
  Request.RequestAddHeader(builder, header)
  Request.RequestAddPayload(builder, payload)
  builder.Finish(Request.RequestEnd(builder))
  return builder.Output()


def RequestFrom(buffer_):
  request = Request.Request.GetRootAsRequest(buffer_, 0)
  return RequestDTO(request.Header(), _PayloadFrom(request))


def MakeResponse(dto, builderInitialSize=0):
  builder = flatbuffers.Builder(builderInitialSize)
  status = dto.status
  payload = _CreatePayload(builder, dto.payload)
  Response.ResponseStart(builder)
  Response.ResponseAddStatus(builder, status)
  Response.ResponseAddPayload(builder, payload)
  builder.Finish(Response.ResponseEnd(builder))
  return builder.Output()


def ResponseFrom(buffer_):
  response = Response.Response.GetRootAsResponse(buffer_, 0)
  return ResponseDTO(response.Status(), _PayloadFrom(response))


def _CreatePayload(builder, buffer_):
  Request.RequestStartPayloadVector(builder, len(buffer_))
  for b in reversed(buffer_):
    builder.PrependByte(b)
  return builder.EndVector(len(buffer_))


def _PayloadFrom(flatbuffer):
  return bytes(flatbuffer.Payload(i) for i in range(flatbuffer.PayloadLength()))
