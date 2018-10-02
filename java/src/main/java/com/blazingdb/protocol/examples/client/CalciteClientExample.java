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
    public static void test ()  {
//            byte[] payload = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        DMLRequestMessage requestPayload = new DMLRequestMessage("query");
//            System.out.println(new DMLRequestMessage(requestPayload.getBufferData()).getQuery());
        FlatBufferBuilder builder = new FlatBufferBuilder(0);

        ByteBuffer payloadBuf = requestPayload.getBufferData();
        byte[] payload = payloadBuf.array();
        System.out.println("payload:" );
        System.out.println( Arrays.toString(payload) );

        int payloadOffset = builder.createByteVector(payload);

        // int root = Request.createRequest(builder, headerOffset, payloadOffset);
        Request.startRequest(builder);
        Request.addHeader(builder, Header.createHeader(builder, (byte)1, 2, 3));
        Request.addPayload(builder, payloadOffset);
        int root = Request.endRequest(builder);

        builder.finish(root);
        ByteBuffer buf = builder.dataBuffer();

        Request request = Request.getRootAsRequest(buf);
        Header header = request.header();
        assert header.messageType() == (byte)1;
        assert header.payloadLength() == (long)2;
        assert header.accessToken() == (long)3;

        System.out.println(header.messageType() );
        System.out.println(header.payloadLength() );
        System.out.println(header.accessToken() );

        System.out.println("payload:" );
        ByteBuffer payloadAsByteBuffer = request.payloadAsByteBuffer();
        payloadAsByteBuffer.rewind();


        DMLRequestMessage clonePayload = new DMLRequestMessage(payloadAsByteBuffer);
        System.out.println( clonePayload.getQuery() );
        System.out.println(Arrays.toString(request.payloadAsByteBuffer().array()) );



    }
    public static void test2() {
        IMessage requestPayload = new DMLRequestMessage("query");
        HeaderMessage header = new HeaderMessage(MessageType.DML, 0L, 0L);
        RequestMessage requestObj =  new RequestMessage(header, requestPayload);
        ByteBuffer buf = requestObj.getBufferData();
        Request cloneRequest = Request.getRootAsRequest(buf);

        DMLRequestMessage clonePayload = new DMLRequestMessage(cloneRequest.payloadAsByteBuffer());
        System.out.println(Arrays.toString(cloneRequest.payloadAsByteBuffer().array()) );

        System.out.println( clonePayload.getQuery() );
    }

    public static void test3() {
        DMLRequestMessage requestPayload = new DMLRequestMessage("query3");


        HeaderMessage header = new HeaderMessage(MessageType.DML, 0L, 0L);
        RequestMessage requestObj =  new RequestMessage(header , requestPayload);
        ByteBuffer buf = requestObj.getBufferData();
        RequestMessage serverMessage = new RequestMessage(buf);

        RequestMessage cloneRequest = new RequestMessage(buf);

        DMLRequestMessage clonePayload = new DMLRequestMessage(cloneRequest.getPayloadBuffer());
        System.out.println( clonePayload.getQuery() );
    }
    public static void test4 () {
        DMLRequestMessage requestPayload = new DMLRequestMessage("query4");
        HeaderMessage header = new HeaderMessage(MessageType.DML, 0L, 0L);
        RequestMessage requestObj =  new RequestMessage(header , requestPayload);
        ByteBuffer buf = requestObj.getBufferData();

        System.out.println("send buffer");
        System.out.println(Arrays.toString(ByteBufferUtil.cloneByteBuffer(buf).array()) );
        System.out.println(new DMLRequestMessage(Response.getRootAsResponse(buf).payloadAsByteBuffer()).getQuery());
        System.out.println();

    }

    public static void main(String []args) throws IOException {
//
//        test2();
//        test3();

//        test4();

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