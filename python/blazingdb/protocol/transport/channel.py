import flatbuffers

import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.messages.blazingdb.protocol \
  import Header, Request, Response, ResponseError


class RequestSchema(transport.schema(Request)):
  header = transport.StructSegment(Header)
  payload = transport.BytesSegment()


def MakeAuthRequestBuffer(messageType, schema, builderInitialSize=0):
  return MakeRequestBuffer(messageType, 0, schema, builderInitialSize)


class ResponseSchema(transport.schema(Response)):
  status = transport.NumberSegment()  # todo(gcca): [Enum,Choice]Segment
  payload = transport.BytesSegment()


class ResponseErrorSchema(transport.schema(ResponseError)):
  errors = transport.StringSegment()


def MakeRequestBuffer(messageType, accessToken, schema, builderInitialSize=0):
  payload = schema.ToBuffer()
  request = RequestSchema(header={
    'messageType': messageType,
    'accessToken': accessToken,
  }, payload=payload).ToBuffer()
  return request

def RequestSchemaFrom(buffer_):
  return RequestSchema.From(buffer_)
