package com.blazingdb.protocol.message.calcite;

import blazingdb.protocol.calcite.DDLResponse;
import blazingdb.protocol.orchestrator.DDLCreateTableRequest;
import blazingdb.protocol.orchestrator.SchemaList;

import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;

import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.util.ArrayList;



public class SchemaListMessage implements IMessage {
    private final ArrayList<DDLCreateTableRequestMessage> schemas;

    public SchemaListMessage(ByteBuffer payload) {
        this.schemas = new ArrayList<>();
        SchemaList message = SchemaList.getRootAsSchemaList(payload);
        for (int i = 0; i < message.tablesLength(); i++) {
            DDLCreateTableRequest tableSchema = message.tables(i);
            ArrayList<String> columnNames = new ArrayList<>();
            for (int index = 0; index < tableSchema.columnNamesLength(); index++) {
                columnNames.add(tableSchema.columnNames(index));
            }
            ArrayList<String>  columnTypes = new ArrayList<>();
            for (int index = 0; index < tableSchema.columnTypesLength(); index++) {
                columnTypes.add(tableSchema.columnTypes(index));
            }
            this.schemas.add(new DDLCreateTableRequestMessage(columnNames, columnTypes, tableSchema.name(), tableSchema.dbName()));
        }
    }
    /**
     * @return the schemas
     */
    public ArrayList<DDLCreateTableRequestMessage> getSchemas() {
        return schemas;
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        return builder.dataBuffer();
    }

    @Override
    public long getBufferSize() {
        return 0;
    }
}
