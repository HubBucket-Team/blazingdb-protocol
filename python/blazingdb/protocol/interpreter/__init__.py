import flatbuffers

from blazingdb.protocol.internal import MakeBuffer

import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.messages.blazingdb.protocol.interpreter \
  import GetResultRequest

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage


class GetResultSchema(transport.schema(GetResultRequest)):
  token = transport.StringSegment()
