package com.blazingdb.protocol.ipc.calcite;

import com.blazingdb.protocol.ipc.IMessage;
import com.blazingdb.protocol.ipc.util.ByteBufferUtils;

import java.nio.ByteBuffer;

public class DDLResponseMessage implements IMessage {
    public DDLResponseMessage(ByteBuffer payload) {
    }

    public DDLResponseMessage() {

    }

    @Override
    public ByteBuffer getBufferData() {
        return ByteBufferUtils.addEOF(ByteBuffer.wrap("".getBytes()));
    }
}
