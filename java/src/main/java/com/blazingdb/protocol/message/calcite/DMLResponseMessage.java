package com.blazingdb.protocol.message.calcite;

import blazingdb.protocol.calcite.DMLResponse;
import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public final class DMLResponseMessage implements IMessage {
    private final String logicalPlan;
		private final long time;

    public DMLResponseMessage(final String plan, final long time) {
        this.logicalPlan = plan;
				this.time = time;
    }

    public DMLResponseMessage(ByteBuffer buffer) {
        DMLResponse message = DMLResponse.getRootAsDMLResponse(buffer);
        this.logicalPlan = message.logicalPlan();
				this.time = message.time();
    }

    public String getLogicalPlan() {
        return logicalPlan;
    }

    @Override
    public ByteBuffer getBufferData () {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int string_data = builder.createString(ByteBuffer.wrap(logicalPlan.getBytes(StandardCharsets.US_ASCII)));
        int root = DMLResponse.createDMLResponse(builder, string_data, time);
        builder.finish(root);
        return ByteBufferUtil.addEof(builder.dataBuffer());
    }
    @Override
    public long getBufferSize() {
        return 0;
    }
}

