package com.blazingdb.protocol.examples.simple;

import com.blazingdb.protocol.UnixClient;
import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.message.calcite.DMLRequestMessage;
import com.blazingdb.protocol.message.calcite.DMLResponseMessage;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.TimeUnit;

public class CalciteClientExample {

    public static void main(String []args) throws IOException, InterruptedException {
        File unixSocketFile = new File("/tmp/socket");
        String statement = "select * from orders";
        IMessage request =  new DMLRequestMessage(statement);

        wait(unixSocketFile);

        UnixClient client = new UnixClient(unixSocketFile);
        ByteBuffer result = client.send(request.getBufferData());
        DMLResponseMessage response = new DMLResponseMessage(result);
        System.out.println(response.getLogicalPlan());
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
