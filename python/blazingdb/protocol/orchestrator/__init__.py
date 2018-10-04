import flatbuffers

from blazingdb.protocol.errors import Error
import blazingdb.protocol.transport
import blazingdb.protocol.transport as transport
from blazingdb.messages.blazingdb.protocol.Status import Status
from blazingdb.messages.blazingdb.protocol.ResponseError import ResponseError
from blazingdb.messages.blazingdb.protocol.orchestrator \
    import DMLRequest, DMLResponse, DDLRequest, DDLResponse
from blazingdb.messages.blazingdb.protocol.orchestrator.MessageType \
  import MessageType as OrchestratorMessageType

from blazingdb.messages.blazingdb.protocol.orchestrator \
  import AuthRequest, AuthResponse

class DMLRequestSchema(transport.schema(DMLRequest)):
  query = transport.StringSegment()

class DDLRequestSchema(transport.schema(DDLRequest)):
  query = transport.StringSegment()

class DMLResponseSchema(transport.schema(DMLResponse)):
  resultToken = transport.NumberSegment()

class AuthResponseSchema(transport.schema(AuthResponse)):
  accessToken = transport.NumberSegment()

class DMLRequestDTO:

  def __init__(self, query):
    self.query = query


class DMLResponseDTO:

  def __init__(self, token):
    self.token = token


def MakeDMLRequest(sessionToken, query):
  builder = flatbuffers.Builder(512)
  query = builder.CreateString(query)
  DMLRequest.DMLRequestStart(builder)
  DMLRequest.DMLRequestAddQuery(builder, query)
  builder.Finish(DMLRequest.DMLRequestEnd(builder))
  output = builder.Output()
  return blazingdb.protocol.transport.MakeRequest(
    blazingdb.protocol.transport.RequestDTO(
      blazingdb.protocol.transport.HeaderDTO(
        messageType=OrchestratorMessageType.DML,
        payloadLength=len(output),
        sessionToken=sessionToken),
      output), 512)


def MakeDDLRequest(sessionToken, query):
  builder = flatbuffers.Builder(512)
  query = builder.CreateString(query)
  DDLRequest.DDLRequestStart(builder)
  DDLRequest.DDLRequestAddQuery(builder, query)
  builder.Finish(DMLRequest.DMLRequestEnd(builder))
  output = builder.Output()
  return blazingdb.protocol.transport.MakeRequest(
    blazingdb.protocol.transport.RequestDTO(
      blazingdb.protocol.transport.HeaderDTO(
        messageType=OrchestratorMessageType.DDL,
        payloadLength=len(output),
        sessionToken=sessionToken),
      output), 512)

def DMLRequestFrom(buffer_):
  request = blazingdb.protocol.transport.RequestFrom(buffer_)
  dmlRequest = DMLRequest.DMLRequest.GetRootAsDMLRequest(request.payload, 0)
  return blazingdb.protocol.transport.RequestDTO(
    request.header, DMLRequestDTO(dmlRequest.Query()))


def MakeDMLResponse(token):
  builder = flatbuffers.Builder(512)
  token = builder.CreateString(token)
  DMLResponse.DMLResponseStart(builder)
  DMLResponse.DMLResponseAddResultToken(builder, token)
  builder.Finish(DMLResponse.DMLResponseEnd(builder))
  return blazingdb.protocol.transport.MakeResponse(
    blazingdb.protocol.transport.ResponseDTO(Status.Success,
                                             builder.Output()), 512)


def DMLResponseFrom(buffer_):
  response = blazingdb.protocol.transport.ResponseFrom(buffer_)
  if response.status == Status.Error:
    errorResponse = ResponseError.GetRootAsResponseError(response.payload, 0)
    raise Error(errorResponse.Errors())
  dmlResponse = DMLResponse.DMLResponse.GetRootAsDMLResponse(response.payload, 0)
  return blazingdb.protocol.transport.ResponseDTO(
      response.status, DMLResponseDTO(dmlResponse.ResultToken()))


def DDLResponseFrom(buffer_):
  response = blazingdb.protocol.transport.ResponseFrom(buffer_)
  if response.status == Status.Error:
    errorResponse = ResponseError.GetRootAsResponseError(response.payload, 0)
    raise Error(errorResponse.Errors())
  return blazingdb.protocol.transport.ResponseDTO(response.status, None)


#authorization

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
  if response.status == Status.Error:
    errorResponse = ResponseError.GetRootAsResponseError(response.payload, 0)
    raise Error(errorResponse.Errors())
  authResponse = AuthResponse.AuthResponse.GetRootAsAuthResponse(response.payload, 0)
  return blazingdb.protocol.transport.ResponseDTO(
    response.status, AuthResponseDTO(authResponse.AccessToken()))
