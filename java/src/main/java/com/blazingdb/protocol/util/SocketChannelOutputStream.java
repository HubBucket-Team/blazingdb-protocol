package com.blazingdb.protocol.util;

import jnr.unixsocket.UnixSocketChannel;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class SocketChannelOutputStream extends OutputStream {
    private final UnixSocketChannel ch;

    public SocketChannelOutputStream(UnixSocketChannel ch) {
        this.ch = ch;
    }


    @Override
    public void write(int value) throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(value);
        buffer.rewind();
        ch.write(buffer);
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        ch.write(ByteBuffer.wrap(b, off, len));
    }

    @Override
    public void close() throws IOException {
        this.ch.close();
    }
}
