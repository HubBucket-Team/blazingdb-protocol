package com.blazingdb.protocol.examples.calcite.examples.server;

import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.nio.charset.Charset;

public class ZqmServer {

    static class Server {
        public void run() {
            ZContext ctx = new ZContext();
            ZMQ.Socket  server = ctx.createSocket(ZMQ.REP);
            server.bind("ipc:///tmp/aocsa");
            while (true){
                String value = server.recvStr();
                server.send("hi back!");
            }

        }
    }
    public static void main (String[] args) {
        Server s = new Server();
        s.run();
    }
}
