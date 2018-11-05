package com.blazingdb.protocol.message.calcite;

import blazingdb.protocol.calcite.DDLResponse;

import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;

import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;

public class DDLResponseMessage implements IMessage {
		private final long time;

    public DDLResponseMessage(ByteBuffer payload) {
				DDLResponse message = DDLResponse.getRootAsDDLResponse(payload);
				this.time = message.time();
    }

    public DDLResponseMessage(final long time) {
				this.time = time;
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int root = DDLResponse.createDDLResponse(builder, time);
        builder.finish(root);
        return ByteBufferUtil.addEof(builder.dataBuffer());
    }
    @Override
    public long getBufferSize() {
        return 0;
    }
}
