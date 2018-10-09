package com.blazingdb.protocol.message.calcite;

import blazingdb.protocol.flatbuf.calcite.DDLCreateTableRequest;
import blazingdb.protocol.flatbuf.calcite.DDLRequest;
import blazingdb.protocol.flatbuf.calcite.DataType;

import com.blazingdb.protocol.message.IMessage;
import com.blazingdb.protocol.util.ByteBufferUtil;
import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public final class DDLCreateTableRequestMessage implements IMessage {
    
	private List<String> columnNames;
	private List<String> types;
	private String name;
	private String dbName;
	
	public String getDbName() {
		return dbName;
	}
	public String getName() {
		return name;
	}
	
	public List<String> getColumnNames(){
		return columnNames;
	}
	
	public List<String> getColumnTypes(){
		return types;
	}
	
    public DDLCreateTableRequestMessage(final List<String> columnNames, final List<String> types) {
        this.columnNames = columnNames;
        this.types = types;
    }

    public DDLCreateTableRequestMessage(ByteBuffer buffer) {
        DDLCreateTableRequest message = DDLCreateTableRequest.getRootAsDDLCreateTableRequest(buffer);
        this.columnNames = new ArrayList<String>();
        this.types = new ArrayList<String>();
        if(message.columnNamesLength() != message.columnTypesLength()) {
    	   //TODO: have some kind of error
       }
       for(int i = 0; i < message.columnNamesLength(); i++) {
    	   this.columnNames.add(message.columnNames(i));
    	   this.types.add(message.columnTypes(i));
       }
       this.name = message.name();
       this.dbName = message.dbName();
    }


    @Override
    public ByteBuffer getBufferData() {
        FlatBufferBuilder builder = new FlatBufferBuilder(1024);
        int tableNameOffset = builder.createString(this.name);
        int[] columnNameOffsets = new int[columnNames.size()];
        int[] columnTypeOffsets = new int[types.size()];
        
        for(int i =0; i < columnNames.size(); i++) {
        	int curColumn = builder.createString(columnNames.get(i));
        	columnNameOffsets[i] = curColumn;
        	curColumn = builder.createString(types.get(i));
        	columnTypeOffsets[i] = curColumn;
        }
        
        int root = DDLCreateTableRequest.createDDLCreateTableRequest(
        		builder, 
        		tableNameOffset, 
        		DDLCreateTableRequest.createColumnNamesVector(builder, columnNameOffsets), 
        		DDLCreateTableRequest.createColumnTypesVector(builder,columnTypeOffsets),
        		builder.createString(dbName));
        
        builder.finish(root);
        return ByteBufferUtil.addEof(builder.dataBuffer());
    }
    @Override
    public long getBufferSize() {
        return 0;
    }
}

