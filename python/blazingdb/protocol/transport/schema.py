"""Schema and Segment for flatbuffer objects transporting."""

import abc

import flatbuffers


class MetaSchema(type):
  """Metaclass for Schema.

  To fix up segment members of the concrete schema.
  """

  def __init__(cls, name, bases, classdict):
    super(MetaSchema, cls).__init__(name, bases, classdict)
    cls._fix_up_segments()


class SchemaAttribute(abc.ABC):
  """A base class to identify schema members."""

  @abc.abstractmethod
  def _fix_up(self, cls, name):
    return NotImplemented


class Schema(metaclass=MetaSchema):
  """A class describing Flatbuffers schema.

  All classes inheriting from Schema have MetaSchema, so that segments
  are fixed up after class definition.

  class ConcreteSchema(Schema):
    _module = flatbuffer.user.message.module

    field_str = StringSegment()
    field_int = NumberSegment()

  The related flatbuffers module is `None` by default. You must override
  in parent scope in order to have working `ToBuffer` and `From` methods. See
  `schema` function as a tool to create concrete schema classes.
  """

  _segments = None
  _module = None
  _values = None
  _nested = None
  _inline = None

  def ToBuffer(self):
    builder = flatbuffers.Builder(0)

    pairs = []
    for segment in self._nested:
      pairs.append((segment._name, segment._bytes(builder, self)))

    module = self._module
    name = module.__name__.split('.')[-1]
    getattr(module, name + 'Start')(builder)

    for segment in self._inline:
      pairs.append((segment._name, segment._bytes(builder, self)))

    for member, value in reversed(pairs):
      getattr(module, '%sAdd%s' % (name, member.capitalize()))(builder, value)
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
  def From(cls, buffer):
    module = cls._module
    name = module.__name__.split('.')[-1]
    obj = getattr(getattr(module, name), 'GetRootAs' + name)(buffer, 0)
    members = {name: segment._from(obj)
               for name, segment in cls._segments.items()}
    name = cls.__name__.split('.')[-1]
    return type(name[0].lower() + name[1:], (), members)


  @classmethod
  def _fix_up_segments(cls):
    if __name__ == cls.__module__:
      return
    cls._segments = {}
    cls._nested = []
    cls._inline = []
    for name in set(dir(cls)):
      attr = getattr(cls, name, None)
      if isinstance(attr, SchemaAttribute):
        attr._fix_up(cls, name)
        if isinstance(attr, Segment):
          cls._segments[name] = attr
          if isinstance(attr, Nested):
            cls._nested.append(attr)
          elif isinstance(attr, Inline):
            cls._inline.append(attr)
          else:
            raise TypeError('Bad `%s` segment type' % name)


class Segment(SchemaAttribute):

  _name = None

  def _fix_up(self, cls, name):
    self._name = name

  @abc.abstractmethod
  def _bytes(self, builder, schema):
    return NotImplemented

  @abc.abstractmethod
  def _from(self, obj):
    return NotImplemented

  def _set_value(self, schema, value):
    schema._values[self._name] = value


class Nested:
  """Mark for segments with data for inside flatbuffers objs."""


class Inline:
  """Mark for segments with inline data for flatbuffers objs."""


class NumberSegment(Segment, Inline):

  def _bytes(self, builder, schema):
    return schema._values[self._name]

  def _from(self, obj):
    return getattr(obj, self._name.capitalize())()


class StringSegment(Segment, Nested):

  def _bytes(self, builder, schema):
    return builder.CreateString(schema._values[self._name])

  def _from(self, obj):
    return getattr(obj, self._name.capitalize())()


class BytesSegment(Segment, Nested):

  def _bytes(self, builder, schema):
    module = schema._module
    name = module.__name__.split('.')[-1]
    member = self._name.capitalize()
    buffer = schema._values[self._name]
    getattr(module, '%sStart%sVector' % (name, member))(builder, len(buffer))
    for byte in reversed(buffer):
      builder.PrependByte(byte)
    return builder.EndVector(len(buffer))

  def _from(self, obj):
    name = self._name.capitalize()
    byte = getattr(obj, name)
    return bytes(byte(i) for i in range(getattr(obj, name + 'Length')()))


class StructSegment(Segment, Inline):

  def __init__(self, module):
    self._module = module

  def _bytes(self, builder, schema):
    module = self._module
    name = module.__name__.split('.')[-1]
    value = schema._values[self._name]
    return getattr(module, 'Create' + name)(builder, **value)

  def _from(self, obj):
    struct = getattr(obj, self._name.capitalize())()
    members = {name[0].lower() + name[1:]: getattr(struct, name)()
               for name in set(dir(struct)) - set(('Init', ))
               if name[0].isalpha()}
    return type(self._name, (), members)
