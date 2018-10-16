package com.blazingdb.protocol.util;

import jnr.unixsocket.UnixSocketChannel;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class SocketChannelInputStream extends InputStream {

    private final UnixSocketChannel ch;

    public SocketChannelInputStream(UnixSocketChannel ch) {
        this.ch = ch;
    }

    @Override
    public int read() throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        ch.read(buffer);
        buffer.rewind();
        int size = buffer.getInt();
        return size;
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        return ch.read(ByteBuffer.wrap(b, off, len));
    }
    @Override
    public void close() throws IOException {
        this.ch.close();
    }
}