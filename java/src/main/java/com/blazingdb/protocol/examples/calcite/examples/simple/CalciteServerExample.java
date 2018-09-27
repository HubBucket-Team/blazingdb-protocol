package com.blazingdb.protocol.examples.calcite.examples.simple;

import com.blazingdb.protocol.IService;
import com.blazingdb.protocol.UnixService;
import com.blazingdb.protocol.ipc.calcite.DMLRequestMessage;
import com.blazingdb.protocol.ipc.calcite.DMLResponseMessage;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class CalciteServerExample {
	public static void main(String[] args) throws IOException {
        File unixSocketFile = new File("/tmp/socket");
        unixSocketFile.deleteOnExit();

        IService calciteService  = new IService() {
            @Override
            public ByteBuffer process(ByteBuffer buffer) {
                DMLRequestMessage request = new DMLRequestMessage(buffer);
                System.out.println("##ByteBuffer statement_:" + request.getQuery());

                String logicalPlan = "LogicalUnion(all=[false])\n" +
                        "  LogicalUnion(all=[false])\n" +
                        "    LogicalProject(EXPR$0=[$1], join_x=[$0])\n" +
                        "      LogicalAggregate(group=[{0}], EXPR$0=[SUM($1)])\n" +
                        "        LogicalProject(join_x=[$4], join_x0=[$7])\n" +
                        "          LogicalJoin(condition=[=($7, $0)], joinType=[inner])\n";

                DMLResponseMessage response = new DMLResponseMessage(logicalPlan);
                return response.getBufferData();
            }
        };
		UnixService service = new UnixService(calciteService);
        service.bind(unixSocketFile);
        new Thread(service).start();
	}
}
