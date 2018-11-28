package com.blazingdb.protocol.calcite.plan;

import com.blazingdb.protocol.calcite.plan.messages.RexNode;
import com.blazingdb.protocol.calcite.plan.messages.RexNodeType;

import org.apache.commons.lang3.Validate;

import com.google.flatbuffers.FlatBufferBuilder;

import java.util.Collection;

final class RexNodeFactory {

  private final FlatBufferBuilder flatBufferBuilder;

  public RexNodeFactory() { flatBufferBuilder = new FlatBufferBuilder(0); }

  public Integer createRexCallNodeOffset(final SqlKind sqlKind,
                                         final SqlTypeName sqlTypeName,
                                         final Collection<Integer> operands) {
    Validate.notEmpty(operands, "Call operands are required");
    return createRexNodeOffset(RexNodeType.Call,
                               (short) sqlKind.ordinal(),
                               (short) sqlTypeName.ordinal(),
                               0);
  }

  protected Integer createRexNodeOffset(final short rexNodeType,
                                        final short sqlKind,
                                        final short sqlTypeName,
                                        final       byte[] data) {
    final Integer dataOffset =
        RexNode.createDataVector(flatBufferBuilder, data);
    return createRexNodeOffset(rexNodeType, sqlKind, sqlTypeName, dataOffset);
  }

  protected Integer createRexNodeOffset(final short rexNodeType,
                                        final short sqlKind,
                                        final short sqlTypeName,
                                        final Integer dataOffset) {
    return RexNode.createRexNode(
        flatBufferBuilder, rexNodeType, sqlKind, sqlTypeName, dataOffset);
  }
}
