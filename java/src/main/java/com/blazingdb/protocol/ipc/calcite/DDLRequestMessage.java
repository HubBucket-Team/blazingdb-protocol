package com.blazingdb.protocol.ipc.calcite;

import blazingdb.protocol.calcite.DMLRequest;
import com.blazingdb.protocol.ipc.IMessage;
import com.blazingdb.protocol.ipc.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public final class DDLRequestMessage implements IMessage {
    private final String query;

    public DDLRequestMessage(final String query) {
        this.query = query;
    }

    public DDLRequestMessage(ByteBuffer buffer) {
        DMLRequest message = DMLRequest.getRootAsDMLRequest(buffer);
        this.query = message.query();
    }

    public String getQuery() {
        return query;
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int statement_string_data = builder.createString(ByteBuffer.wrap(query.getBytes(StandardCharsets.US_ASCII)));
        int root = DMLRequest.createDMLRequest(builder, statement_string_data);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }
}

