package com.blazingdb.protocol.message;

import blazingdb.protocol.Header;
import blazingdb.protocol.Request;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.util.Arrays;

public class RequestMessage implements IMessage {

    HeaderMessage header;
    ByteBuffer payloadBuffer;

    public RequestMessage(HeaderMessage header, IMessage requestPayload) {
        this.header = header;
        this.payloadBuffer = requestPayload.getBufferData();
    }

    public ByteBuffer getPayloadBuffer() {
        return payloadBuffer;
    }

    public short getHeaderType() {
        return header.messageType;
    }

    public RequestMessage(ByteBuffer buf) {
        Request  pointer = Request.getRootAsRequest(buf);
        Header headerPointer =  pointer.header();
        payloadBuffer =  pointer.payloadAsByteBuffer();

        this.header = new HeaderMessage(headerPointer.messageType(), headerPointer.accessToken());
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(0);
        int payloadOffset = builder.createByteVector(payloadBuffer.array());
        Request.startRequest(builder);
        Request.addHeader(builder, Header.createHeader(builder, (byte)header.messageType, (long)header.accessToken));
        Request.addPayload(builder, payloadOffset);
        int root = Request.endRequest(builder);
        builder.finish(root);
        return builder.dataBuffer();
    }

    @Override
    public long getBufferSize() {
        return 0;
    }
}
