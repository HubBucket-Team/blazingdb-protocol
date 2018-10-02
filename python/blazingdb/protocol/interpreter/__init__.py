import flatbuffers

import blazingdb.protocol.transport as transport
import blazingdb.protocol.transport

from blazingdb.messages.blazingdb.protocol.Status import Status

from blazingdb.messages.blazingdb.protocol.interpreter \
  import GetResultRequest

from blazingdb.messages.blazingdb.protocol.interpreter.MessageType \
  import MessageType as InterpreterMessage


class GetResultSchema(transport.schema(GetResultRequest)):
  token = transport.StringSegment()


def AuthRequestFrom(buffer_):
  request = blazingdb.protocol.transport.RequestFrom(buffer_)
  dmlRequest = DMLRequest.DMLRequest.GetRootAsDMLRequest(request.payload, 0)
  return blazingdb.protocol.transport.RequestDTO(
    request.header, DMLRequestDTO(dmlRequest.Query()))
