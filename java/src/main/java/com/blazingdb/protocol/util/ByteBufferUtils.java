package com.blazingdb.protocol.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class ByteBufferUtils {
    public static ByteBuffer addEOF(ByteBuffer byteBuffer) {
        byte[] bytesArray = new byte[byteBuffer.remaining()];
        byteBuffer.get(bytesArray, 0, bytesArray.length);
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream( );
        try {
            outputStream.write( bytesArray );
            outputStream.write( new String("\n").getBytes() );
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ByteBuffer.wrap(outputStream.toByteArray());
    }
}
