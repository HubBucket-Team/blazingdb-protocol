package com.blazingdb.protocol.message.calcite;

import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;

import java.nio.ByteBuffer;

public class DDLResponseMessage implements IMessage {
    public DDLResponseMessage(ByteBuffer payload) {
    }

    public DDLResponseMessage() {

    }

    @Override
    public ByteBuffer getBufferData() {
        return ByteBufferUtil.addEof(ByteBuffer.wrap("".getBytes()));
    }
    @Override
    public long getBufferSize() {
        return 0;
    }
}
