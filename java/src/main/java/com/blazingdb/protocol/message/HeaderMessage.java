package com.blazingdb.protocol.message;

public class HeaderMessage {
    public HeaderMessage(byte messageType, long accessToken) {
        this.messageType = messageType;
        this.accessToken = accessToken;
    }

    public byte messageType;
    public long accessToken;
}
