package com.blazingdb.protocol.message;

public class HeaderMessage {
    public HeaderMessage(byte messageType, long payloadLength, long accessToken) {
        this.messageType = messageType;
        this.payloadLength = payloadLength;
        this.accessToken = accessToken;
    }

    public byte messageType;
    public long payloadLength;
    public long accessToken;
}
