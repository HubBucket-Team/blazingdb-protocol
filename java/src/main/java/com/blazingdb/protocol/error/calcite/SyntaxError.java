package com.blazingdb.protocol.error.calcite;

import com.blazingdb.protocol.error.Error;

public class SyntaxError extends Error {
    public SyntaxError(String message) {
        super(message);
    }
}
