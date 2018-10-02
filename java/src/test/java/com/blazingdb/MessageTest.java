package com.blazingdb;

import blazingdb.protocol.Header;
import blazingdb.protocol.Request;
import blazingdb.protocol.Response;
import blazingdb.protocol.calcite.MessageType;
import com.blazingdb.protocol.message.HeaderMessage;
import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.message.RequestMessage;
import com.blazingdb.protocol.message.calcite.DMLRequestMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;
import com.google.flatbuffers.FlatBufferBuilder;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * Unit test for simple App.
 */
public class MessageTest
    extends TestCase
{
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public MessageTest(String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( MessageTest.class );
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
    /**
     * Rigourous Test :-)
     */
    public void testApp()
    {
        test();
        test2();
        test3();
        test4();


        assertTrue( true );
    }
}
