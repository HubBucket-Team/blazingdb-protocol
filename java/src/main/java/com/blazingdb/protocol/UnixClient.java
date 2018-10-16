package com.blazingdb.protocol;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;

import com.blazingdb.protocol.util.ByteBufferUtil;
import com.blazingdb.protocol.util.SocketChannelInputStream;
import com.blazingdb.protocol.util.SocketChannelOutputStream;
import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;

public class UnixClient {
    private final File unixSocket;
    UnixSocketAddress address = null;
    UnixSocketChannel channel = null;

    public UnixClient(File unixSocket) throws IOException {
        this.unixSocket = unixSocket;
        try {
            address = new UnixSocketAddress(this.unixSocket);
            channel = UnixSocketChannel.open(address);
        } catch (Throwable e) {
            final String error = "ERROR: Service seems offline, could not use the unix connection socket "
                    + this.unixSocket.getAbsolutePath();
            System.err.println(error);
            if (channel != null) {
                channel.close();
            }
            channel = null;
            throw new IOException(error);
        }
    }

    public byte[] send(byte[] message) {
        OutputStream out = new SocketChannelOutputStream(channel);
        byte[] result = null;
        try {
            out.write(message.length);
            out.write(message);
            SocketChannelInputStream inputStream = new SocketChannelInputStream(channel);
            int length = inputStream.read();
            result = new byte[length];
            inputStream.read(result);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return result;
    }
    public ByteBuffer send(ByteBuffer message) {
        return ByteBuffer.wrap(send(ByteBufferUtil.getByteArrayFromByteBuffer(message).array()));
    }
} 
