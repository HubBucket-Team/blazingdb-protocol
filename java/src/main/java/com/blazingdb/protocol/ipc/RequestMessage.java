package com.blazingdb.protocol.ipc;

import blazingdb.protocol.Header;
import blazingdb.protocol.Request;
import com.blazingdb.protocol.ipc.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;

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

    public RequestMessage(ByteBuffer buffer) {
        Request message = Request.getRootAsRequest(buffer);
        Header tmpHeader = message.header();
        this.header.messageType = tmpHeader.messageType();
        this.header.payloadLength = tmpHeader.payloadLength();
        this.header.sessionToken = tmpHeader.sessionToken();

        this.payloadBuffer = message.payloadAsByteBuffer();
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int payloadOffset = builder.createByteVector(payloadBuffer.array());
//        int root = Request.createRequest(builder, header,  payloadOffset);
        Request.startRequest(builder);
        Request.addHeader(builder, Header.createHeader(builder, header.messageType, header.payloadLength, header.sessionToken));
        Request.addPayload(builder, payloadOffset);
        int root = Request.endRequest(builder);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }
}
