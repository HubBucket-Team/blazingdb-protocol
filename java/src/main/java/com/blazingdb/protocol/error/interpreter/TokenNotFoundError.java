package com.blazingdb.protocol.error.interpreter;

import com.blazingdb.protocol.error.Error;

public class TokenNotFoundError extends Error {
    public TokenNotFoundError(String message) {
        super(message);
    }
}
