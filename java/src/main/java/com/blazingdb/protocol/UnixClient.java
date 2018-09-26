package com.blazingdb.protocol;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;

import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;

public class UnixClient {
    private final File unixSocket;
    UnixSocketAddress address = null;
    UnixSocketChannel channel = null;
    static int MAX_BUFFER_SIZE = 4096;

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

    public byte[] send(byte[] message) throws IOException {
        OutputStream out = Channels.newOutputStream(channel);
        out.write(message);
        out.flush();

        InputStream inputStream = Channels.newInputStream(channel);
        byte[] result = new byte[MAX_BUFFER_SIZE];
        DataInputStream inStream = new DataInputStream(new BufferedInputStream(inputStream));
        inStream.read(result);
        return result;
    }
} 
