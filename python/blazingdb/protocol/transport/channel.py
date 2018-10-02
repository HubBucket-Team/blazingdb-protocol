import flatbuffers

import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.messages.blazingdb.protocol \
  import Header, Request, Response, ResponseError


class RequestSchema(transport.schema(Request)):
  header = transport.StructSegment(Header)
  payload = transport.BytesSegment()


class ResponseSchema(transport.schema(Response)):
  status = transport.NumberSegment()  # todo(gcca): [Enum,Choice]Segment
  payload = transport.BytesSegment()


class ResponseErrorSchema(transport.schema(ResponseError)):
  errors = transport.StringSegment()


def MakeRequestBuffer(messageType, accessToken, schema, builderInitialSize=0):
  payload = schema.ToBuffer()
  return RequestSchema(header={
    'messageType': messageType,
    'payloadLength': len(payload),
    'accessToken': accessToken,
  }, payload=payload).ToBuffer()


def RequestSchemaFrom(buffer_):
  return RequestSchema.From(buffer_)
