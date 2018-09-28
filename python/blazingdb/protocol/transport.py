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
  payload = _MakePayload(builder, dto.payload)
  Request.RequestStart(builder)
  Request.RequestAddHeader(builder, header)
  Request.RequestAddPayload(builder, payload)
  builder.Finish(Request.RequestEnd(builder))
  return builder.Output()


def RequestFrom(buffer_):
  request = Request.Request.GetRootAsRequest(buffer_, 0)
  return RequestDTO(request.Header(), _PayloadFrom(request))


def _MakePayload(builder, buffer_):
  Request.RequestStartPayloadVector(builder, len(buffer_))
  for b in reversed(buffer_):
    builder.PrependByte(b)
  return builder.EndVector(len(buffer_))


def _PayloadFrom(request):
  return bytes(request.Payload(i) for i in range(request.PayloadLength()))
