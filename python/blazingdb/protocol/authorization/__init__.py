import blazingdb.protocol.transport as transport
import blazingdb.protocol.transport
from blazingdb.messages.blazingdb.protocol.authorization \
  import AuthRequest, AuthResponse

from blazingdb.messages.blazingdb.protocol.authorization.MessageType \
  import MessageType as AuthorizationMessage


class AuthRequestSchema(transport.schema(AuthRequest)):
  pass


class AuthResponseSchema(transport.schema(AuthResponse)):
  accessToken = transport.NumberSegment()


class AuthRequestDTO:
  def __init__(self):
    pass


class AuthResponseDTO:
  def __init__(self, accessToken):
    self.accessToken = accessToken


def AuthRequestFrom(buffer_):
  request = blazingdb.protocol.transport.RequestFrom(buffer_)
  authRequest = AuthRequest.AuthRequest.GetRootAsAuthRequest(request.payload, 0)
  return blazingdb.protocol.transport.RequestDTO(
    request.header, AuthRequestDTO())


def AuthResponseFrom(buffer_):
  response = blazingdb.protocol.transport.ResponseFrom(buffer_)
  authResponse = AuthResponse.AuthResponse.GetRootAsAuthResponse(response.payload, 0)
  return blazingdb.protocol.transport.ResponseDTO(
    response.status, AuthResponseDTO(authResponse.AccessToken()))
