package com.blazingdb.protocol.examples.server;

import blazingdb.protocol.Request;
import blazingdb.protocol.Response;
import blazingdb.protocol.Status;
import blazingdb.protocol.calcite.MessageType;
import com.blazingdb.protocol.IService;
import com.blazingdb.protocol.UnixService;
import com.blazingdb.protocol.message.RequestMessage;
import com.blazingdb.protocol.message.ResponseErrorMessage;
import com.blazingdb.protocol.message.ResponseMessage;
import com.blazingdb.protocol.message.calcite.*;
import com.blazingdb.protocol.util.ByteBufferUtil;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class CalciteServiceExample {

    public static void main(String[] args) throws IOException {
        File unixSocketFile = new File("/tmp/calcite.socket");
        unixSocketFile.deleteOnExit();

        IService calciteService  = new IService() {
            @Override
            public ByteBuffer process(ByteBuffer buffer) {
                RequestMessage requestMessage = new RequestMessage(buffer);
                if(requestMessage.getHeaderType() == MessageType.DML) {
                    DMLRequestMessage requestPayload = new DMLRequestMessage(requestMessage.getPayloadBuffer());
                    ResponseMessage response = null;
                    System.out.println("DML: " + requestPayload.getQuery());
                    if (requestPayload.getQuery().contains("select")) {
                        String logicalPlan = ""+
                            "LogicalProject(EXPR$0=[>($0, 5)])\n"+
                            "EnumerableTableScan(table=[[main, nation]])";
                        DMLResponseMessage responsePayload = new DMLResponseMessage(logicalPlan, 98765);
                        response = new ResponseMessage(Status.Success, responsePayload.getBufferData());
                    } else {
                        ResponseErrorMessage error = new ResponseErrorMessage("error: it is not a DML query");
                        response = new ResponseMessage(Status.Error, error.getBufferData());
                    }
                    return response.getBufferData();
                }
                else if(requestMessage.getHeaderType() == MessageType.DDL_CREATE_TABLE) {

                    DDLCreateTableRequestMessage requestPayload = new DDLCreateTableRequestMessage(requestMessage.getPayloadBuffer());
                    ResponseMessage response = null;
                    System.out.println("DDL Create Table: " + requestPayload.getName());
                    System.out.println("\tdbName: " + requestPayload.getDbName());
                    System.out.println("\tColumnNames: " + requestPayload.getColumnNames());
                    System.out.println("\tColumnTypes: " + requestPayload.getColumnTypes());

                    if (requestPayload.getDbName().contains("main") ){
                        DDLResponseMessage responsePayload = new DDLResponseMessage(98765);
                        response = new ResponseMessage(Status.Success, responsePayload.getBufferData());
                    } else {
                        ResponseErrorMessage error = new ResponseErrorMessage("error: it is not a valid DDL Create Table Request");
                        response = new ResponseMessage(Status.Error, error.getBufferData());
                    }
                    return response.getBufferData();
                }
                else if(requestMessage.getHeaderType() == MessageType.DDL_DROP_TABLE) {

                    DDLDropTableRequestMessage requestPayload = new DDLDropTableRequestMessage(requestMessage.getPayloadBuffer());
                    ResponseMessage response = null;
                    System.out.println("DDL Drop Table: " + requestPayload.getName());
                    System.out.println("\tdbName: " + requestPayload.getDbName());

                    if (requestPayload.getDbName().contains("main") ){
                        DDLResponseMessage responsePayload = new DDLResponseMessage(98765);
                        response = new ResponseMessage(Status.Success, responsePayload.getBufferData());
                    } else {
                        ResponseErrorMessage error = new ResponseErrorMessage("error: it is not a valid DDL Drop Table Request");
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
