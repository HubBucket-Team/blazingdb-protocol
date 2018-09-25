package com.blazingdb.protocol.examples;
import com.blazingdb.protocol.Client;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class ExampleClient {
	public final static int BUFFER_SIZE = 4096 * 1024 * 4;
    public final static String PERMISSION_LABEL = "┌∩┐(x_x)┌∩┐";
    public final static String OFFLINE_MESSAGE = "The BlazingDB server is restarting please try again in a moment.";

    public final static String PERMISSIONS_DELIM = "(***BlznPerm***)";
 
    public static void main(String []args) throws IOException, InterruptedException {
        java.io.File path = new java.io.File("/tmp/socket");


        String statement = "select * from orders";
		String authorization = ExampleClient.PERMISSIONS_DELIM;
		
		BlazingQueryMessage message = new BlazingQueryMessage (statement, authorization);
//		BlazingQueryMessage message_clone = new BlazingQueryMessage (message.getBufferData());


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
		Client client = new Client(path);
//		client.send(message.getBufferData());
		 
		
    }
}
