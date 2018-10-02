package com.blazingdb.protocol.error.interpreter;

import com.blazingdb.protocol.error.Error;

public class LogicalPlanError extends Error {
    public LogicalPlanError(String message) {
        super(message);
    }
}
