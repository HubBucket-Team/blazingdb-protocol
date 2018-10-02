package com.blazingdb.protocol.error;

import java.lang.Exception;

public class Error extends Exception {

    private String message;

    public Error (final String message) {
        super(message);
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}
