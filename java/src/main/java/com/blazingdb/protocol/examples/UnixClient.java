package com.blazingdb.protocol.examples;
 
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.channels.Channels;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.IOUtils;

import jnr.unixsocket.UnixSocketAddress;
import jnr.unixsocket.UnixSocketChannel;

public class UnixClient {
	private static byte[] getByteArrayFromByteBuffer(ByteBuffer byteBuffer) {
	    byte[] bytesArray = new byte[byteBuffer.remaining()];
	    byteBuffer.get(bytesArray, 0, bytesArray.length);
	    
	    
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream( );
		try {
			outputStream.write( bytesArray );
			outputStream.write( new String("\n").getBytes() );
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return outputStream.toByteArray( );
	}
 

	public static void main(String[] args) throws IOException, InterruptedException {
        java.io.File path = new java.io.File("/tmp/socket");
        int retries = 0;
        while (!path.exists()) {
            TimeUnit.MILLISECONDS.sleep(500L);
            retries++;
            if (retries > 10) {
                throw new IOException(
                    String.format(
                        "File %s does not exist after retry",
                        path.getAbsolutePath()
                    )
                );
            }
        }
//        String data = "123456789";
        String statement = "select * from orders";
 		String authorization = ExampleClient.PERMISSIONS_DELIM;
     		
 		BlazingQueryMessage queryMessage = new BlazingQueryMessage (statement, authorization);
 		ByteBuffer buffer = queryMessage.getBufferData();
  		byte[] message = getByteArrayFromByteBuffer(buffer);
 		
        UnixSocketAddress address = new UnixSocketAddress(path);
        UnixSocketChannel channel = UnixSocketChannel.open(address);
        System.out.println("connected to " + channel.getRemoteSocketAddress());
        System.out.println("sending ... " + message.toString() + "|sz: " + message.length );
    	System.out.println("arrays: ");
        System.out.println(Arrays.toString(message));

//        PrintWriter out = new PrintWriter(Channels.newOutputStream(channel), true);
//        out.print(message.toString() + "\r\n");
//        out.flush();
    	try {
            OutputStream out = Channels.newOutputStream(channel);
            out.write(message);
            out.flush();
    	} catch (Exception e) {
			// TODO: handle exception
    		e.printStackTrace();
		}
        
        InputStream inputStream = Channels.newInputStream(channel);
//        CharBuffer result = CharBuffer.allocate(1024);
//        inputStream.read(result);
        byte []result = new byte [1024];
        DataInputStream inStream = new DataInputStream
        		(new BufferedInputStream(inputStream));
        
        inStream.read(result);

        System.out.println("read from server: " + result.length + " - " +  Arrays.toString(result).contains( Arrays.toString(message) ));
        System.out.println(Arrays.toString(result));

        final int status;

        {
	    	System.out.println("EQUAL: data match");
	
	    	BlazingQueryMessage serverMessage = new BlazingQueryMessage(result);
	
			System.out.println("## SQL Statement: " + serverMessage.statement() + "  |  with grank/revoke => "
					+ serverMessage.authorization());
			System.out.println("## The FlatBuffer was successfully created and verified!");
	
	        System.out.println("SUCCESS");
	        status = 0;
        }
        System.exit(status);
    }
}