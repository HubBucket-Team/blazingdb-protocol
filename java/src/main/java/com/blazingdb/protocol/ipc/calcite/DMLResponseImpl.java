package com.blazingdb.protocol.ipc.calcite;

import blazingdb.protocol.calcite.DMLResponse;
import com.blazingdb.protocol.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public final class DMLResponseImpl {
    private final String logicalPlan;

    public DMLResponseImpl(final String plan) {
        this.logicalPlan = plan;
    }
    public DMLResponseImpl (ByteBuffer buffer) {
        this(buffer.array());
    }

    public String getLogicalPlan() {
        return logicalPlan;
    }

    public DMLResponseImpl(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.rewind();
        DMLResponse message = DMLResponse.getRootAsDMLResponse(buffer);
        this.logicalPlan = message.logicalPlan();
    }

    public ByteBuffer getBufferData () {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int string_data = builder.createString(ByteBuffer.wrap(logicalPlan.getBytes(StandardCharsets.US_ASCII)));
        int root = DMLResponse.createDMLResponse(builder, string_data);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }
}

