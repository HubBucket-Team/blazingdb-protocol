# automatically generated by the FlatBuffers compiler, do not modify

# namespace: orchestrator

import flatbuffers

class DDLCreateTableRequest(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsDDLCreateTableRequest(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DDLCreateTableRequest()
        x.Init(buf, n + offset)
        return x

    # DDLCreateTableRequest
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DDLCreateTableRequest
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # DDLCreateTableRequest
    def ColumnNames(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # DDLCreateTableRequest
    def ColumnNamesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DDLCreateTableRequest
    def ColumnTypes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # DDLCreateTableRequest
    def ColumnTypesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DDLCreateTableRequest
    def DbName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # DDLCreateTableRequest
    def SchemaType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DDLCreateTableRequest
    def Gdf(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .BlazingTable import BlazingTable
            obj = BlazingTable()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # DDLCreateTableRequest
    def Files(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # DDLCreateTableRequest
    def FilesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # DDLCreateTableRequest
    def CsvDelimiter(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # DDLCreateTableRequest
    def CsvLineTerminator(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # DDLCreateTableRequest
    def CsvSkipRows(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DDLCreateTableRequest
    def ResultToken(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

def DDLCreateTableRequestStart(builder): builder.StartObject(11)
def DDLCreateTableRequestAddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def DDLCreateTableRequestAddColumnNames(builder, columnNames): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(columnNames), 0)
def DDLCreateTableRequestStartColumnNamesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def DDLCreateTableRequestAddColumnTypes(builder, columnTypes): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(columnTypes), 0)
def DDLCreateTableRequestStartColumnTypesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def DDLCreateTableRequestAddDbName(builder, dbName): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(dbName), 0)
def DDLCreateTableRequestAddSchemaType(builder, schemaType): builder.PrependInt8Slot(4, schemaType, 0)
def DDLCreateTableRequestAddGdf(builder, gdf): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(gdf), 0)
def DDLCreateTableRequestAddFiles(builder, files): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(files), 0)
def DDLCreateTableRequestStartFilesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def DDLCreateTableRequestAddCsvDelimiter(builder, csvDelimiter): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(csvDelimiter), 0)
def DDLCreateTableRequestAddCsvLineTerminator(builder, csvLineTerminator): builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(csvLineTerminator), 0)
def DDLCreateTableRequestAddCsvSkipRows(builder, csvSkipRows): builder.PrependInt32Slot(9, csvSkipRows, 0)
def DDLCreateTableRequestAddResultToken(builder, resultToken): builder.PrependUint64Slot(10, resultToken, 0)
def DDLCreateTableRequestEnd(builder): return builder.EndObject()
