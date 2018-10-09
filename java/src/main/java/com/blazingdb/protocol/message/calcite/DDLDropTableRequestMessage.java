package com.blazingdb.protocol.message.calcite;


import blazingdb.protocol.calcite.DDLDropTableRequest;


import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;

public final class DDLDropTableRequestMessage implements IMessage {
    

	private String name;
	private String dbName;
	
	public String getDbName() {
		return dbName;
	}
	public String getName() {
		return name;
	}
	
	
    public DDLDropTableRequestMessage(String name, String dbName) {

        this.name = name;
        this.dbName = dbName;
    }

    public DDLDropTableRequestMessage(ByteBuffer buffer) {
        DDLDropTableRequest message = DDLDropTableRequest.getRootAsDDLDropTableRequest(buffer);
    
    
       this.name = message.name();
       this.dbName = message.dbName();
    }


    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);

      
       
        int root = DDLDropTableRequest.createDDLDropTableRequest(
        		builder, 
        		builder.createString(this.name), 
        		builder.createString(dbName));
        
        builder.finish(root);
        return ByteBufferUtil.addEof(builder.dataBuffer());
    }
    @Override
    public long getBufferSize() {
        return 0;
    }
}

