package com.blazingdb.protocol;

import java.nio.ByteBuffer;

public interface IService {
    ByteBuffer process(ByteBuffer buffer);
}
