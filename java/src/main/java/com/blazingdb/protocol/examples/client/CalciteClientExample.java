package com.blazingdb.protocol.examples.client;

import blazingdb.protocol.Header;
import blazingdb.protocol.Request;
import blazingdb.protocol.Response;
import blazingdb.protocol.Status;
import blazingdb.protocol.calcite.MessageType;
import com.blazingdb.protocol.UnixClient;
import com.blazingdb.protocol.error.calcite.SyntaxError;
import com.blazingdb.protocol.message.*;
import com.blazingdb.protocol.message.calcite.DDLResponseMessage;
import com.blazingdb.protocol.message.calcite.DMLRequestMessage;
import com.blazingdb.protocol.message.calcite.DMLResponseMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;
import com.google.flatbuffers.FlatBufferBuilder;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class CalciteClientExample {

    static class CalciteClient {
        File unixSocketFile;
        UnixClient client;

        CalciteClient() throws IOException {
            unixSocketFile = new File("/tmp/calcite.socket");
            client = new UnixClient(unixSocketFile);
        }


        public String getLogicalPlan(String query) throws SyntaxError {
            DMLRequestMessage requestPayload = new DMLRequestMessage(query);
            HeaderMessage header = new HeaderMessage(MessageType.DML, 0L, 0L);
            RequestMessage requestObj =  new RequestMessage(header , requestPayload);
            ByteBuffer buf = requestObj.getBufferData();
            ByteBuffer result =  client.send(buf);
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
