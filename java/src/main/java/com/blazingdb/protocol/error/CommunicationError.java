package com.blazingdb.protocol.error;

public class CommunicationError extends Error {
    public CommunicationError(String message) {
        super(message);
    }
}
