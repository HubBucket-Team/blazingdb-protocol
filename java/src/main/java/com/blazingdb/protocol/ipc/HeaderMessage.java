package com.blazingdb.protocol.ipc;

public class HeaderMessage {
    public HeaderMessage(byte messageType, long payloadLength, long sessionToken) {
        this.messageType = messageType;
        this.payloadLength = payloadLength;
        this.sessionToken = sessionToken;
    }

    byte messageType;
    long payloadLength;
    long sessionToken;
}
