package com.blazingdb.protocol.examples.calcite.examples.client;

import blazingdb.protocol.Header;
import blazingdb.protocol.Status;
import com.blazingdb.protocol.UnixClient;
import com.blazingdb.protocol.error.calcite.SyntaxError;
import com.blazingdb.protocol.message.calcite.DDLResponseMessage;
import com.blazingdb.protocol.message.calcite.DMLRequestMessage;
import com.blazingdb.protocol.message.calcite.DMLResponseMessage;
import com.blazingdb.protocol.message.*;
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

        public String getLogicalPlan(String query) throws SyntaxError {
            IMessage requestPayload = new DMLRequestMessage(query);

            //@todo: byte messageType, long payloadLength, long sessionToken
            HeaderMessage header = new HeaderMessage(MessageType.DML, 0L, 0L);
            RequestMessage requestObj =  new RequestMessage(header , requestPayload);
            ByteBuffer result = client.send(requestObj.getBufferData());
            ResponseMessage response = new ResponseMessage(result);
            if (response.getStatus() == Status.Error) {
                ResponseErrorMessage responsePayload = new ResponseErrorMessage(response.getPayload());
                throw new SyntaxError(responsePayload.getError());
            }
            DMLResponseMessage responsePayload = new DMLResponseMessage(response.getPayload());
            return responsePayload.getLogicalPlan();
        }

        public byte updateSchema(String query) throws SyntaxError {
            IMessage requestPayload = new DMLRequestMessage(query);
            HeaderMessage header = new HeaderMessage(MessageType.DDL, 0L, 0L);

            RequestMessage requestObj =  new RequestMessage(header, requestPayload);
            ByteBuffer result = client.send(requestObj.getBufferData());
            ResponseMessage response = new ResponseMessage(result);
            if (response.getStatus() == Status.Error) {
                ResponseErrorMessage responsePayload = new ResponseErrorMessage(response.getPayload());
                throw new SyntaxError(responsePayload.getError());
            }
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
            } catch (SyntaxError error) {
                System.out.println(error.getMessage());
            }

            String statementDDL = "create database alexbd";
            try {
                byte status = client.updateSchema(statementDDL);
                System.out.println(status);
            } catch (SyntaxError error) {
                System.out.println(error.getMessage());
            }
        }
        {
            String statement = "celect * from orders";
            try {
                String logicalPlan = client.getLogicalPlan(statement);
                System.out.println(logicalPlan);
            } catch (SyntaxError error) {
                System.out.println(error.getMessage());
            }

            String statementDDL = "treate database alexbd";
            try {
                byte status = client.updateSchema(statementDDL);
                System.out.println(status);
            } catch (SyntaxError error) {
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