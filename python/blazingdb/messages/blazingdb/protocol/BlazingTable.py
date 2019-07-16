# automatically generated by the FlatBuffers compiler, do not modify

# namespace: protocol

import flatbuffers

class BlazingTable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBlazingTable(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BlazingTable()
        x.Init(buf, n + offset)
        return x

    # BlazingTable
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BlazingTable
    def Columns(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from blazingdb.messages.blazingdb.protocol.gdf.gdf_column_handler import gdf_column_handler
            obj = gdf_column_handler()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # BlazingTable
    def ColumnsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # BlazingTable
    def ColumnTokens(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # BlazingTable
    def ColumnTokensAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint64Flags, o)
        return 0

    # BlazingTable
    def ColumnTokensLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # BlazingTable
    def ResultToken(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

def BlazingTableStart(builder): builder.StartObject(3)
def BlazingTableAddColumns(builder, columns): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(columns), 0)
def BlazingTableStartColumnsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def BlazingTableAddColumnTokens(builder, columnTokens): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(columnTokens), 0)
def BlazingTableStartColumnTokensVector(builder, numElems): return builder.StartVector(8, numElems, 8)
def BlazingTableAddResultToken(builder, resultToken): builder.PrependUint64Slot(2, resultToken, 0)
def BlazingTableEnd(builder): return builder.EndObject()
