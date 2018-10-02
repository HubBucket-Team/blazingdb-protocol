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

        this.header = new HeaderMessage(headerPointer.messageType(), headerPointer.payloadLength(), headerPointer.accessToken());
     }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(0);
        int payloadOffset = builder.createByteVector(payloadBuffer.array());
        Request.startRequest(builder);
        Request.addHeader(builder, Header.createHeader(builder, (byte)header.messageType, (long)header.payloadLength, (long)header.accessToken));
        Request.addPayload(builder, payloadOffset);
        int root = Request.endRequest(builder);
        builder.finish(root);
        return builder.dataBuffer();
    }

//    byte[] payload = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//
//    FlatBufferBuilder builder = new FlatBufferBuilder(0);
//    int payloadOffset = builder.createByteVector(payload);
//
//        Request.startRequest(builder);
//        Request.addHeader(builder, Header.createHeader(builder, (byte)1, 2, 3));
//        Request.addPayload(builder, payloadOffset);
//    int root = Request.endRequest(builder);
//
//        builder.finish(root);
//    ByteBuffer buf = builder.dataBuffer();
//
//    Request request = Request.getRootAsRequest(buf);
//        assert request.header().messageType() == (byte)1;
//        assert request.header().payloadLength() == (long)2;
//        assert request.header().accessToken() == (long)3;

    @Override
    public long getBufferSize() {
        return 0;
    }
}
