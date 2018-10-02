package com.blazingdb.protocol.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class ByteBufferUtil {

    public static ByteBuffer getByteArrayFromByteBuffer(ByteBuffer byteBuffer) {
        byte[] bytesArray = new byte[byteBuffer.remaining()];
        byteBuffer.get(bytesArray, 0, bytesArray.length);


        ByteArrayOutputStream outputStream = new ByteArrayOutputStream( );
        try {
            outputStream.write( bytesArray );
            outputStream.write( "\n".getBytes() );
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return ByteBuffer.wrap(outputStream.toByteArray( ));
    }

    public static ByteBuffer addEof(ByteBuffer byteBuffer) {
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

    public static ByteBuffer deepCopyVisible( ByteBuffer orig )
    {
        int pos = orig.position();
        try
        {
            ByteBuffer toReturn;
            // try to maintain implementation to keep performance
            if( orig.isDirect() )
                toReturn = ByteBuffer.allocateDirect(orig.remaining());
            else
                toReturn = ByteBuffer.allocate(orig.remaining());

            toReturn.put(orig);
            toReturn.order(orig.order());

            return (ByteBuffer) toReturn.position(0);
        }
        finally
        {
            orig.position(pos);
        }
    }

    public static ByteBuffer cloneByteBuffer( ByteBuffer orig )
    {
        int pos = orig.position(), lim = orig.limit();
        try
        {
            orig.position(0).limit(orig.capacity()); // set range to entire buffer
            ByteBuffer toReturn = deepCopyVisible(orig); // deep copy range
            toReturn.position(pos).limit(lim); // set range to original
            return toReturn;
        }
        finally // do in finally in case something goes wrong we don't bork the orig
        {
            orig.position(pos).limit(lim); // restore original
        }
    }

    public static ByteBuffer concat(final ByteBuffer... buffers) {
        final ByteBuffer combined = ByteBuffer.allocate(Arrays.stream(buffers).mapToInt(Buffer::remaining).sum());
        Arrays.stream(buffers).forEach(b -> combined.put(b.duplicate()));
        return combined;
    }

    public static ByteBuffer addSizeToPrefix(ByteBuffer buffer) {
        ByteBuffer sizeBuffer = ByteBuffer.allocate(4);
        sizeBuffer.putInt(buffer.array().length);

        System.out.println("requestSize: ");
        sizeBuffer.rewind();
        buffer.rewind();
        ByteBuffer concatResult = concat(sizeBuffer, buffer);

        if (sizeBuffer.remaining() < 4) {
            System.out.println(
                    "Remaining buffer too short to contain length of length-prefixed field"
                            + ". Remaining: " + sizeBuffer.remaining());
        }
        System.out.println(sizeBuffer.getInt());
        concatResult.rewind();
        System.out.println(concatResult.getInt());

        return concatResult;
    }
}
