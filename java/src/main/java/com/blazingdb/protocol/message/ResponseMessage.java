package com.blazingdb.protocol.message;

import blazingdb.protocol.Response;
import com.blazingdb.protocol.util.ByteBufferUtil;
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
        return ByteBufferUtil.addEof(builder.dataBuffer());
    }

    @Override
    public long getBufferSize() {
        return 0;
    }

    public byte getStatus() {
        return status;
    }

    public ByteBuffer getPayload() {
        return payload;
    }
}
