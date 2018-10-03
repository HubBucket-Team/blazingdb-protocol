import flatbuffers

import blazingdb.protocol.transport as transport
import blazingdb.protocol.transport

from blazingdb.messages.blazingdb.protocol.interpreter \
  import DMLRequest, DMLResponse, GetResultRequest, GetResultResponse

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage


class DMLRequestSchema(transport.schema(DMLRequest)):
  logicalPlan = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.StringSegment()


class GetResultRequestSchema(transport.schema(GetResultRequest)):
  token = transport.StringSegment()


class GetResultResponseSchema(transport.schema(GetResultResponse)):
  names = transport.StringSegment()
  values = transport.StringSegment()
