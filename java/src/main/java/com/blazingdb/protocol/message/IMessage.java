package com.blazingdb.protocol.message;

import java.nio.ByteBuffer;

public interface IMessage {

    ByteBuffer getBufferData();

    long getBufferSize();
}
