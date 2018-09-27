package com.blazingdb.protocol.examples.calcite.examples.server;

import blazingdb.protocol.HeaderType;
import blazingdb.protocol.Status;
import com.blazingdb.protocol.IService;
import com.blazingdb.protocol.UnixService;
import com.blazingdb.protocol.ipc.calcite.DDLRequestMessage;
import com.blazingdb.protocol.ipc.calcite.DDLResponseMessage;
import com.blazingdb.protocol.ipc.calcite.DMLRequestMessage;
import com.blazingdb.protocol.ipc.calcite.DMLResponseMessage;
import com.blazingdb.protocol.ipc.RequestMessage;
import com.blazingdb.protocol.ipc.ResponseErrorMessage;
import com.blazingdb.protocol.ipc.ResponseMessage;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class CalciteServiceExample {

    public static void main(String[] args) throws IOException {
        File unixSocketFile = new File("/tmp/socket");
        unixSocketFile.deleteOnExit();

        IService calciteService  = new IService() {
            @Override
            public ByteBuffer process(ByteBuffer buffer) {
                RequestMessage requestMessage = new RequestMessage(buffer);

                if(requestMessage.getHeaderType() == HeaderType.DML) {
                    DMLRequestMessage requestPayload = new DMLRequestMessage(requestMessage.getPayloadBuffer());
                    ResponseMessage response = null;
                    if (requestPayload.getQuery().contains("select")) {
                        String logicalPlan = logicalPlan = "LogicalUnion(all=[false])\n" +
                                "  LogicalUnion(all=[false])\n" +
                                "    LogicalProject(EXPR$0=[$1], join_x=[$0])\n" +
                                "      LogicalAggregate(group=[{0}], EXPR$0=[SUM($1)])\n" +
                                "        LogicalProject(join_x=[$4], join_x0=[$7])\n" +
                                "          LogicalJoin(condition=[=($7, $0)], joinType=[inner])\n";
                        DMLResponseMessage responsePayload = new DMLResponseMessage(logicalPlan);
                        response = new ResponseMessage(Status.Success, responsePayload.getBufferData());
                    } else {
                        ResponseErrorMessage error = new ResponseErrorMessage("error: it is not a DML query");
                        response = new ResponseMessage(Status.Error, error.getBufferData());
                    }
                    return response.getBufferData();
                }
                else if(requestMessage.getHeaderType() == HeaderType.DDL) {
                    DDLRequestMessage requestPayload = new DDLRequestMessage(requestMessage.getPayloadBuffer());
                    ResponseMessage response = null;
                    if (requestPayload.getQuery().contains("create")){
                        DDLResponseMessage responsePayload = new DDLResponseMessage();
                        response = new ResponseMessage(Status.Success, responsePayload.getBufferData());
                    } else {
                        ResponseErrorMessage error = new ResponseErrorMessage("error: it is not a DDL query");
                        response = new ResponseMessage(Status.Error, error.getBufferData());
                    }
                    return response.getBufferData();
                }
                return null;
            }
        };
        UnixService service = new UnixService(calciteService);
        service.bind(unixSocketFile);
        new Thread(service).start();
    }
}
