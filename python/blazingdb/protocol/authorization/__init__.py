import blazingdb.protocol.transport as transport

from blazingdb.messages.blazingdb.protocol.authorization \
  import AuthRequest, AuthResponse

from blazingdb.messages.blazingdb.protocol.authorization.MessageType \
  import MessageType as AuthorizationMessage


class AuthRequestSchema(transport.schema(AuthRequest)):
  pass


class AuthResponseSchema(transport.schema(AuthResponse)):
  accessToken = transport.NumberSegment()
