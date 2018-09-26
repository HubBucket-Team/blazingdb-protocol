package com.blazingdb.protocol.ipc.calcite;
import blazingdb.protocol.calcite.DMLRequest;
import com.blazingdb.protocol.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public final class DMLRequestImpl{
    private final String query;

    public DMLRequestImpl(final String query) {
        this.query = query;
    }
    public DMLRequestImpl (ByteBuffer buffer) {
        this(buffer.array());
    }

    public String getQuery() {
        return query;
    }

    public DMLRequestImpl(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.rewind();

        DMLRequest message = DMLRequest.getRootAsDMLRequest(buffer);
        this.query = message.query();
    }

    public ByteBuffer getBufferData () {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int statement_string_data = builder.createString(ByteBuffer.wrap(query.getBytes(StandardCharsets.US_ASCII)));
        int root = DMLRequest.createDMLRequest(builder, statement_string_data);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }
}

