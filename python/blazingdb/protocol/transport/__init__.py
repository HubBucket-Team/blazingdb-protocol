import abc

import flatbuffers

from blazingdb.messages.blazingdb.protocol import Header, Request, Response


__all__ = (
  'HeaderDTO',
  'RequestDTO',
  'ResponseDTO',
  'MakeRequest',
  'RequestFrom',
)


class HeaderDTO:

  def __init__(self, messageType, payloadLength, sessionToken):
    self.messageType = messageType
    self.payloadLength = payloadLength
    self.sessionToken = sessionToken


class RequestDTO:

  def __init__(self, header, payload):
    self.header = header
    self.payload = payload


class ResponseDTO:

  def __init__(self, status, payload):
    self.status = status
    self.payload = payload


def MakeRequest(dto, builderInitialSize=0):
  builder = flatbuffers.Builder(builderInitialSize)
  payload = _CreatePayload(builder, dto.payload)
  Request.RequestStart(builder)
  headerDto = dto.header
  header = Header.CreateHeader(builder,
    headerDto.messageType, headerDto.payloadLength, headerDto.sessionToken)
  Request.RequestAddHeader(builder, header)
  Request.RequestAddPayload(builder, payload)
  builder.Finish(Request.RequestEnd(builder))
  return builder.Output()


def RequestFrom(buffer_):
  request = Request.Request.GetRootAsRequest(buffer_, 0)
  return RequestDTO(request.Header(), _PayloadFrom(request))


def MakeResponse(dto, builderInitialSize=0):
  builder = flatbuffers.Builder(builderInitialSize)
  status = dto.status
  payload = _CreatePayload(builder, dto.payload)
  Response.ResponseStart(builder)
  Response.ResponseAddStatus(builder, status)
  Response.ResponseAddPayload(builder, payload)
  builder.Finish(Response.ResponseEnd(builder))
  return builder.Output()


def ResponseFrom(buffer_):
  response = Response.Response.GetRootAsResponse(buffer_, 0)
  return ResponseDTO(response.Status(), _PayloadFrom(response))


def _CreatePayload(builder, buffer_):
  Request.RequestStartPayloadVector(builder, len(buffer_))
  for b in reversed(buffer_):
    builder.PrependByte(b)
  return builder.EndVector(len(buffer_))


def _PayloadFrom(flatbuffer):
  return bytes(flatbuffer.Payload(i) for i in range(flatbuffer.PayloadLength()))


def schema(module):
  return type(module.__name__ + 'SchemaBase', (Schema,), dict(_module=module))


class MetaSchema(type):

  def __init__(cls, name, bases, classdict):
    super(MetaSchema, cls).__init__(name, bases, classdict)
    cls._fix_up_segments()


class SchemaAttribute(abc.ABC):

  @abc.abstractmethod
  def _fix_up(self, cls, code_name):
    return NotImplemented


class Schema(metaclass=MetaSchema):

  _module = None
  _segments = None
  _values = None

  def ToBuffer(self):
    builder = flatbuffers.Builder(0)

    pairs = []
    for segment in self._segments.values():
      name = segment._name
      pairs.append((name, segment._bytes(builder, self._values[name])))

    module = self._module
    name = module.__name__.split('.')[-1]
    getattr(module, name + 'Start')(builder)
    for k, v in pairs:
      getattr(module, '%sAdd%s' % (name, k.capitalize()))(builder, v)
    builder.Finish(getattr(module, name + 'End')(builder))

    return builder.Output()

  def __init__(self, **kargs):
    self._values = {}
    self._set_attributes(kargs)

  def _set_attributes(self, kargs):
    cls = self.__class__
    for name, value in kargs.items():
      segment = getattr(cls, name)
      if not isinstance(segment, Segment):
        raise TypeError('Non segment %s' % name)
      segment._set_value(self, value)

  @classmethod
  def _fix_up_segments(cls):
    cls._segments = {}
    if __name__ == cls.__module__:
      return
    for name in set(dir(cls)):
      attr = getattr(cls, name, None)
      if isinstance(attr, SchemaAttribute):
        attr._fix_up(cls, name)
        if isinstance(attr, Segment):
          cls._segments[name] = attr


class Segment(SchemaAttribute):

  def _fix_up(self, cls, name):
    self._name = name

  @abc.abstractmethod
  def _bytes(self, builder, value):
    return NotImplemented

  def _set_value(self, schema, value):
    schema._values[self._name] = value


class NumberSegment(Segment):

  @staticmethod
  def _bytes(builder, value):
    return value


class StringSegment(Segment):

  @staticmethod
  def _bytes(builder, value):
    return builder.CreateString(value)
