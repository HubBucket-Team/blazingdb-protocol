package com.blazingdb.protocol.examples;

import com.blazingdb.protocol.IService;
import com.blazingdb.protocol.UnixService;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class SimpleServerExample {

	public static void main(String[] args) throws IOException {
        java.io.File unixSocketFile = new File("/tmp/socket");
        unixSocketFile.deleteOnExit();

        IService toUpperService = new IService() {
            @Override
            public ByteBuffer process(ByteBuffer buffer) {
                System.out.printf("Read at pos %d   \n", buffer.position());
                String upper = new String(buffer.array()).toUpperCase();
                System.out.println("server reply: " + upper);
                return ByteBuffer.wrap(upper.getBytes());
            }
        };
		UnixService service = new UnixService(toUpperService);
		service.bind(unixSocketFile);

        new Thread(service).start();
	}
}
