import flatbuffers

from blazingdb.protocol.errors import Error
import blazingdb.protocol.transport
import blazingdb.protocol.transport as transport
from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.messages.blazingdb.protocol.ResponseError import ResponseError
from blazingdb.messages.blazingdb.protocol.orchestrator \
    import DMLRequest, DMLResponse, DDLRequest, DDLResponse, DDLCreateTableRequest
from blazingdb.messages.blazingdb.protocol.orchestrator.MessageType \
  import MessageType as OrchestratorMessageType

from blazingdb.messages.blazingdb.protocol.orchestrator \
  import AuthRequest, AuthResponse

class DMLRequestSchema(transport.schema(DMLRequest)):
  query = transport.StringSegment()

class DDLRequestSchema(transport.schema(DDLRequest)):
  query = transport.StringSegment()

class DDLCreateTableRequestSchema(transport.schema(DDLCreateTableRequest)):
  name = transport.StringSegment()
  columnNames = transport.VectorStringSegment(transport.StringSegment)
  columnTypes = transport.VectorStringSegment(transport.StringSegment)
  dbName = transport.StringSegment()


class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.NumberSegment()

class AuthResponseSchema(transport.schema(AuthResponse)):
  accessToken = transport.NumberSegment()

class AuthRequestSchema(transport.schema(AuthRequest)):
  pass
