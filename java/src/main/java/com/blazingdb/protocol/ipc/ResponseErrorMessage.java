package com.blazingdb.protocol.ipc;

import blazingdb.protocol.ResponseError;
import com.blazingdb.protocol.ipc.IMessage;
import com.blazingdb.protocol.ipc.util.ByteBufferUtils;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

public class ResponseErrorMessage implements IMessage {

    private String error;

    public ResponseErrorMessage(String error) {
        this.error = error;
    }
    public ResponseErrorMessage(ByteBuffer buffer) {
        this.error = ResponseError.getRootAsResponseError(buffer).errors();
    }

    public String getError() {
        return error;
    }

    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int statement_string_data = builder.createString(ByteBuffer.wrap(error.getBytes(StandardCharsets.US_ASCII)));
        int root = ResponseError.createResponseError(builder, statement_string_data);
        builder.finish(root);
        return ByteBufferUtils.addEOF(builder.dataBuffer());
    }
}
