import flatbuffers

from blazingdb.messages.blazingdb.protocol import Header, Request, Response


def MakeRequestBuffer(messageType, sessionToken, schema, builderInitialSize=0):
  builder = flatbuffers.Builder(builderInitialSize)
  payload, payloadLength = _CreatePayload(builder, schema.ToBuffer())
  Request.RequestStart(builder)
  header = Header.CreateHeader(builder,
    messageType, payloadLength, sessionToken)
  Request.RequestAddHeader(builder, header)
  Request.RequestAddPayload(builder, payload)
  builder.Finish(Request.RequestEnd(builder))
  return builder.Output()


def _CreatePayload(builder, buffer_):
  length = len(buffer_)
  Request.RequestStartPayloadVector(builder, length)
  for b in reversed(buffer_):
    builder.PrependByte(b)
  return builder.EndVector(length), length
