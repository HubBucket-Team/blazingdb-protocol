# automatically generated by the FlatBuffers compiler, do not modify

# namespace: interpreter

import flatbuffers

class RegisterDaskSliceResponse(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRegisterDaskSliceResponse(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RegisterDaskSliceResponse()
        x.Init(buf, n + offset)
        return x

    # RegisterDaskSliceResponse
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RegisterDaskSliceResponse
    def Table(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from blazingdb.messages.blazingdb.protocol.BlazingTable import BlazingTable
            obj = BlazingTable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def RegisterDaskSliceResponseStart(builder): builder.StartObject(1)
def RegisterDaskSliceResponseAddTable(builder, table): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(table), 0)
def RegisterDaskSliceResponseEnd(builder): return builder.EndObject()
