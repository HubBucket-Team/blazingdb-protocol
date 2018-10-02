package com.blazingdb.protocol.examples.calcite.examples;
import com.blazingdb.protocol.UnixClient;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class SimpleClientExample{

    public static void main(String []args) throws IOException, InterruptedException {
        File unixSocketFile = new File("/tmp/socket");
        String statement = "select * from orders\r\n";

        wait(unixSocketFile);

        UnixClient client = new UnixClient(unixSocketFile);
        System.out.println("send message : " + statement);
		byte [] result = client.send(statement.getBytes());
        System.out.println("receive reply: " + new String(result));

    }

    static void wait(File unixSocketFile) throws InterruptedException, IOException {
        int retries = 0;
        while (!unixSocketFile.exists()) {
            TimeUnit.MILLISECONDS.sleep(500L);
            retries++;
            if (retries > 5) {
                throw new IOException(
                    String.format(
                        "File %s does not exist after retry",
                        unixSocketFile.getAbsolutePath()
                    )
                );
            }
        }
    }
}