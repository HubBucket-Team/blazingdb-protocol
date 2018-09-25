package com.blazingdb.protocol;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.channels.Channels;

import org.apache.commons.io.IOUtils;

import com.blazingdb.protocol.examples.BlazingQueryMessage;

import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;

public class Client {
    private final File unixSocket;
    UnixSocketAddress address = null;
    UnixSocketChannel channel = null;
    
    public Client(File unixSocket) throws IOException {
        this.unixSocket = unixSocket;

        try {
            
        } catch (Throwable e) {
            final String error = "ERROR: BlazingDB seems offline, could not use the unix connection socket "
                    + this.unixSocket.getAbsolutePath();
            System.err.println(error);

            if (channel != null) {
                channel.close();
            }

            throw new IOException(error);
        }
    }

    public void send(byte[] bytes)  throws IOException{

    	PrintWriter out = null;
    	address = new UnixSocketAddress(this.unixSocket);
        channel = UnixSocketChannel.open(address);

        System.out.println("connected to " + channel.getRemoteSocketAddress());
        
    	if (channel != null) {
			out = new PrintWriter(Channels.newOutputStream(channel), true);
			 

 	    	System.out.println("##sending buffer:" +  new String(bytes));

			System.out.println("connected to " + channel.getRemoteSocketAddress());
	        PrintWriter w = new PrintWriter(Channels.newOutputStream(channel));
	        w.print(bytes);

	        InputStreamReader r = new InputStreamReader(Channels.newInputStream(channel));
	        CharBuffer result = CharBuffer.allocate(1024);
	        r.read(result);
	        result.flip();
	        System.out.println("read from server: " + result.toString());
	        final int status;
	        if (!result.toString().equals(bytes)) {
	            System.out.println("ERROR: data mismatch");
	            status = -1;
	        } else {
	        	 ByteBuffer byteResult = ByteBuffer.allocate(1024);
	        	    CharBuffer converter = byteResult.asCharBuffer();
	        	    converter.append(result);
	        	BlazingQueryMessage serverMessage = new BlazingQueryMessage(byteResult.array());

				System.out.println("## SQL Statement: " + serverMessage.statement() + "  |  with grank/revoke => "
						+ serverMessage.authorization());
				System.out.println("## The FlatBuffer was successfully created and verified!");

	            System.out.println("SUCCESS");
	            status = 0;
	        }
	        System.exit(status);
		} 
    }
} 
