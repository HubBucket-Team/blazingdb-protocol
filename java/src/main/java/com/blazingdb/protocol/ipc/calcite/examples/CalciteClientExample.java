package com.blazingdb.protocol.ipc.calcite.examples;
import com.blazingdb.protocol.UnixClient;
import com.blazingdb.protocol.ipc.calcite.DMLRequestImpl;
import com.blazingdb.protocol.ipc.calcite.DMLResponseImpl;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class CalciteClientExample {

    public static void main(String []args) throws IOException, InterruptedException {
        java.io.File unixSocketFile = new File("/tmp/socket");
        String statement = "select * from orders";
        DMLRequestImpl request =  new DMLRequestImpl(statement);

        wait(unixSocketFile);

        UnixClient client = new UnixClient(unixSocketFile);
        byte[] result = client.send(request.getBufferData().array());
        DMLResponseImpl response = new DMLResponseImpl(result);
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
