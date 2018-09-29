package com.blazingdb.protocol.examples.calcite.examples.client;

import blazingdb.protocol.Status;
import com.blazingdb.protocol.UnixClient;
import com.blazingdb.protocol.ipc.calcite.DDLResponseMessage;
import com.blazingdb.protocol.ipc.calcite.DMLRequestMessage;
import com.blazingdb.protocol.ipc.calcite.DMLResponseMessage;
import com.blazingdb.protocol.ipc.*;
import blazingdb.protocol.calcite.MessageType;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;

public class CalciteClientExample {

    static class CalciteClient {
        File unixSocketFile;
        UnixClient client;

        CalciteClient() throws IOException {
            unixSocketFile = new File("/tmp/calcite.socket");
            client = new UnixClient(unixSocketFile);
        }

        public String getLogicalPlan(String query) throws IpcException {
            IMessage requestPayload = new DMLRequestMessage(query);
            RequestMessage requestObj =  new RequestMessage(MessageType.DML, requestPayload);
            ByteBuffer result = client.send(requestObj.getBufferData());
            ResponseMessage response = new ResponseMessage(result);
            if (response.getStatus() == Status.Error) {
                ResponseErrorMessage responsePayload = new ResponseErrorMessage(response.getPayload());
                throw new IpcException(responsePayload.getError());
            }
            DMLResponseMessage responsePayload = new DMLResponseMessage(response.getPayload());
            return responsePayload.getLogicalPlan();
        }

        public byte updateSchema(String query) throws IpcException {
            IMessage requestPayload = new DMLRequestMessage(query);
            RequestMessage requestObj =  new RequestMessage(MessageType.DDL, requestPayload);
            ByteBuffer result = client.send(requestObj.getBufferData());
            ResponseMessage response = new ResponseMessage(result);
            if (response.getStatus() == Status.Error) {
                ResponseErrorMessage responsePayload = new ResponseErrorMessage(response.getPayload());
                throw new IpcException(responsePayload.getError());
            }
            DDLResponseMessage responsePayload = new DDLResponseMessage(response.getPayload());
            return response.getStatus();
        }
    }

    public static void main(String []args) throws IOException {
        CalciteClient client = new CalciteClient();
        {
            String statement = "selects * from orders";
            try {
                String logicalPlan = client.getLogicalPlan(statement);
                System.out.println(logicalPlan);
            } catch (IpcException error) {
                System.out.println(error.getMessage());
            }

            String statementDDL = "create database alexbd";
            try {
                byte status = client.updateSchema(statementDDL);
                System.out.println(status);
            } catch (IpcException error) {
                System.out.println(error.getMessage());
            }
        }
        {
            String statement = "celect * from orders";
            try {
                String logicalPlan = client.getLogicalPlan(statement);
                System.out.println(logicalPlan);
            } catch (IpcException error) {
                System.out.println(error.getMessage());
            }

            String statementDDL = "treate database alexbd";
            try {
                byte status = client.updateSchema(statementDDL);
                System.out.println(status);
            } catch (IpcException error) {
                System.out.println(error.getMessage());
            }
        }
    }
}

//todo : tests
//        IMessage requestPayload = new DMLRequestMessage(statement);
//        RequestMessage requestObj =  new RequestMessage(HeaderType.DML, requestPayload);
//
//        RequestMessage cloneRequest = new RequestMessage(requestObj.getBufferData());
//
//        System.out.println(cloneRequest.getHeaderType());
//
//        DMLRequestMessage clonePayload = new DMLRequestMessage(cloneRequest.getPayloadBuffer());
//        System.out.println( clonePayload.getQuery() );