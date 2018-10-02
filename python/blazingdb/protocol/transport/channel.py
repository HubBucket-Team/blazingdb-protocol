import flatbuffers

import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol import Header, Request, Response


class RequestSchema(transport.schema(Request)):
  header = transport.StructSegment(Header)
  payload = transport.BytesSegment()


def MakeRequestBuffer(messageType, accessToken, schema, builderInitialSize=0):
  payload = schema.ToBuffer()
  return RequestSchema(header={
    'messageType': messageType,
    'payloadLength': len(payload),
    'accessToken': accessToken,
  }, payload=payload).ToBuffer()


def RequestSchemaFrom(buffer_):
  return RequestSchema.From(buffer_)
