# automatically generated by the FlatBuffers compiler, do not modify

# namespace: calcite

import flatbuffers

class DMLRequest(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDMLRequest(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DMLRequest()
        x.Init(buf, n + offset)
        return x

    # DMLRequest
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DMLRequest
    def Query(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def DMLRequestStart(builder): builder.StartObject(1)
def DMLRequestAddQuery(builder, query): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(query), 0)
def DMLRequestEnd(builder): return builder.EndObject()
