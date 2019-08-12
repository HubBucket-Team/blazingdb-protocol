# automatically generated by the FlatBuffers compiler, do not modify

# namespace: orchestrator

import flatbuffers

class NodeTable(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNodeTable(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NodeTable()
        x.Init(buf, n + offset)
        return x

    # NodeTable
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # NodeTable
    def Gdf(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from blazingdb.messages.blazingdb.protocol.BlazingTable import BlazingTable
            obj = BlazingTable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def NodeTableStart(builder): builder.StartObject(1)
def NodeTableAddGdf(builder, gdf): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(gdf), 0)
def NodeTableEnd(builder): return builder.EndObject()
