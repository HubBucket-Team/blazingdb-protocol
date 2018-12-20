# automatically generated by the FlatBuffers compiler, do not modify

# namespace: protocol

import flatbuffers

class NodeConnection(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNodeConnection(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NodeConnection()
        x.Init(buf, n + offset)
        return x

    # NodeConnection
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # NodeConnection
    def Path(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # NodeConnection
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def NodeConnectionStart(builder): builder.StartObject(2)
def NodeConnectionAddPath(builder, path): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(path), 0)
def NodeConnectionAddType(builder, type): builder.PrependInt8Slot(1, type, 0)
def NodeConnectionEnd(builder): return builder.EndObject()
