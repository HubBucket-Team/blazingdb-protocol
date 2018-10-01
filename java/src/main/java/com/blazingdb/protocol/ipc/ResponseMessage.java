package com.blazingdb.protocol.ipc;

import blazingdb.protocol.Request;
import blazingdb.protocol.Response;
import com.blazingdb.protocol.ipc.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;

public class ResponseMessage implements IMessage {
    private byte status;
    private ByteBuffer payload;

    public ResponseMessage(byte status, ByteBuffer payload) {
        this.status = status;
        this.payload = payload;
    }

    public ResponseMessage(ByteBuffer buffer) {
        Response message = Response.getRootAsResponse(buffer);
        this.status =  message.status();
        this.payload = message.payloadAsByteBuffer();
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int payloadOffset = builder.createByteVector(payload.array());
        int root = Response.createResponse(builder, status, payloadOffset);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }

    public byte getStatus() {
        return status;
    }

    public ByteBuffer getPayload() {
        return payload;
    }
}
