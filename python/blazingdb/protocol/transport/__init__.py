"""Transport layer to map Flatbuffers modules with user schema classes."""

import flatbuffers

from blazingdb.messages.blazingdb.protocol import Header, Request, Response

from .schema import (Schema,
                     NumberSegment,
                     StringSegment,
                     BytesSegment,
                     StructSegment)


__all__ = (
  'HeaderDTO',
  'RequestDTO',
  'ResponseDTO',
  'MakeRequest',
  'RequestFrom',
)


class HeaderDTO:
  """deprecated."""

  def __init__(self, messageType, payloadLength, sessionToken):
    self.messageType = messageType
    self.payloadLength = payloadLength
    self.sessionToken = sessionToken


class RequestDTO:
  """deprecated."""

  def __init__(self, header, payload):
    self.header = header
    self.payload = payload


class ResponseDTO:
  """deprecated."""

  def __init__(self, status, payload):
    self.status = status
    self.payload = payload


def MakeRequest(dto, builderInitialSize=0):
  """deprecated."""
  builder = flatbuffers.Builder(builderInitialSize)
  payload = _CreatePayload(builder, dto.payload)
  Request.RequestStart(builder)
  headerDto = dto.header
  header = Header.CreateHeader(builder,
                               headerDto.messageType,
                               headerDto.payloadLength,
                               headerDto.sessionToken)
  Request.RequestAddHeader(builder, header)
  Request.RequestAddPayload(builder, payload)
  builder.Finish(Request.RequestEnd(builder))
  return builder.Output()


def RequestFrom(buffer):
  """deprecated."""
  request = Request.Request.GetRootAsRequest(buffer, 0)
  return RequestDTO(request.Header(), _PayloadFrom(request))


def MakeResponse(dto, builderInitialSize=0):
  """deprecated."""
  builder = flatbuffers.Builder(builderInitialSize)
  status = dto.status
  payload = _CreatePayload(builder, dto.payload)
  Response.ResponseStart(builder)
  Response.ResponseAddStatus(builder, status)
  Response.ResponseAddPayload(builder, payload)
  builder.Finish(Response.ResponseEnd(builder))
  return builder.Output()


def ResponseFrom(buffer):
  """deprecated."""
  response = Response.Response.GetRootAsResponse(buffer, 0)
  return ResponseDTO(response.Status(), _PayloadFrom(response))


def _CreatePayload(builder, buffer):
  """deprecated."""
  Request.RequestStartPayloadVector(builder, len(buffer))
  for byte in reversed(buffer):
    builder.PrependByte(byte)
  return builder.EndVector(len(buffer))


def _PayloadFrom(flatbuffer):
  """deprecated."""
  return bytes(flatbuffer.Payload(i) for i in range(flatbuffer.PayloadLength()))


def schema(module):
  """Shortcut to create concrete schema classes.

  Args:
    module: A Flatbuffers module for message

  Useful to have a simple way to define schemas.

  class ConcreteSchema(schema(FlatBuffersModule)):
    field_str = StringSegment()
    field_int = NumberSegment()
  """
  return type(module.__name__ + 'SchemaBase', (Schema,), dict(_module=module))
