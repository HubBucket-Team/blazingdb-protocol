# automatically generated by the FlatBuffers compiler, do not modify

# namespace: interpreter

import flatbuffers

class GetResultResponse(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGetResultResponse(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GetResultResponse()
        x.Init(buf, n + offset)
        return x

    # GetResultResponse
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GetResultResponse
    def Names(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # GetResultResponse
    def NamesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # GetResultResponse
    def Values(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # GetResultResponse
    def ValuesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # GetResultResponse
    def ValuesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def GetResultResponseStart(builder): builder.StartObject(2)
def GetResultResponseAddNames(builder, names): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(names), 0)
def GetResultResponseStartNamesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def GetResultResponseAddValues(builder, values): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(values), 0)
def GetResultResponseStartValuesVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def GetResultResponseEnd(builder): return builder.EndObject()
