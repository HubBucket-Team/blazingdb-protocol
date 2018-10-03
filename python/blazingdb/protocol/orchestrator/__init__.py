import flatbuffers

import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.orchestrator \
    import DMLRequest, DMLResponse

from blazingdb.messages.blazingdb.protocol.orchestrator.MessageType \
  import MessageType as OrchestratorMessageType


class DMLRequestSchema(transport.schema(DMLRequest)):
  query = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.StringSegment()
