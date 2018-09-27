package com.blazingdb.protocol.ipc;

public class IpcException extends Exception{

    private String message;

    public IpcException(String msg){
        super(msg);
        message = msg;
    }


    @Override
    public String getMessage(){
        return message;
    }

}