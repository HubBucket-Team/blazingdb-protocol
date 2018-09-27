package com.blazingdb.protocol.ipc;

import blazingdb.protocol.Request;
import com.blazingdb.protocol.ipc.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.util.Arrays;

public class RequestMessage implements IMessage {

    byte headerType;
    ByteBuffer payloadBuffer;

    public RequestMessage(byte headerType, IMessage requestPayload) {
        this.headerType = headerType;
        this.payloadBuffer = requestPayload.getBufferData();
    }

    public ByteBuffer getPayloadBuffer() {
        return payloadBuffer;
    }

    public short getHeaderType() {
        return headerType;
    }

    public RequestMessage(ByteBuffer buffer) {
        Request message = Request.getRootAsRequest(buffer);
        this.headerType = message.header();
        this.payloadBuffer = message.payloadAsByteBuffer();
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int payloadOffset = builder.createByteVector(payloadBuffer.array());
        int root = Request.createRequest(builder, headerType,  payloadOffset);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }
}
