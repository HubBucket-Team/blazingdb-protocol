package com.blazingdb.protocol.ipc;

public class SyntaxError extends Exception{

    private String message;

    public SyntaxError(String msg){
        super(msg);
        message = msg;
    }


    @Override
    public String getMessage(){
        return message;
    }

}