package com.blazingdb.protocol.examples;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

import com.google.flatbuffers.FlatBufferBuilder;


public class BlazingQueryMessage {

	private String statement_;
	private String authorization_;
	
	public BlazingQueryMessage(String statement, String authorization)
	{
		statement_ = statement;
		authorization_ = authorization;
	}
	      
	public  BlazingQueryMessage(byte[] bytes)
	{
		System.out.println("#bytes read:" + new String(bytes) + "|sz:" + bytes.length);

		java.nio.ByteBuffer buffer = java.nio.ByteBuffer.wrap(bytes);
	    // This convenience method sets the position to 0
		buffer.rewind(); 

     	System.out.println("##dataBuffer position:" + buffer.position());


		

     	System.out.println("##ByteBuffer statement_:" + statement_);

     	System.out.println("##ByteBuffer authorization_:" + authorization_);
		
		
	}
 

	public  BlazingQueryMessage(ByteBuffer buffer)
	{
	    // This convenience method sets the position to 0
		buffer.rewind(); 

      	System.out.println("##ByteBuffer read:" + new String(buffer.array()) + "|sz:" + buffer.array().length);
     	System.out.println("##dataBuffer position:" + buffer.position());


     	System.out.println("##ByteBuffer statement_:" + statement_);

     	System.out.println("##ByteBuffer authorization_:" + authorization_);
     	

	}
	
	public ByteBuffer getBufferData () {
		FlatBufferBuilder builder = new FlatBufferBuilder(1024);
		int statement_string_data = builder.createString(ByteBuffer.wrap(statement_.getBytes(StandardCharsets.US_ASCII)));
		int authorization_string_data = builder.createString(ByteBuffer.wrap(authorization_.getBytes(StandardCharsets.US_ASCII)));

     	System.out.println("##dataBuffer position:" + builder.dataBuffer().position());

		return builder.dataBuffer();
	}
	
	public String authorization() {
		// TODO Auto-generated method stub
		return authorization_;
	}

	public String statement() {
		// TODO Auto-generated method stub
		return statement_;
	}
}