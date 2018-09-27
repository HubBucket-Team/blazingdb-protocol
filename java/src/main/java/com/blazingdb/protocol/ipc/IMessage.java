package com.blazingdb.protocol.ipc;

import java.nio.ByteBuffer;

public interface IMessage {
    ByteBuffer getBufferData();
}
