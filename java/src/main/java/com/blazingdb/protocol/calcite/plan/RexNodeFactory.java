package com.blazingdb.protocol.calcite.plan;

import com.blazingdb.protocol.calcite.plan.messages.RexCall;
import com.blazingdb.protocol.calcite.plan.messages.RexInputRef;
import com.blazingdb.protocol.calcite.plan.messages.RexLiteral;
import com.blazingdb.protocol.calcite.plan.messages.RexNode;
import com.blazingdb.protocol.calcite.plan.messages.RexNodeType;

import org.apache.commons.lang3.Validate;

import com.google.flatbuffers.FlatBufferBuilder;

import java.nio.ByteBuffer;

import java.util.Collection;
import java.util.Optional;

final class RexNodeFactory {

  private final FlatBufferBuilder flatBufferBuilder;

  public RexNodeFactory() { flatBufferBuilder = new FlatBufferBuilder(0); }

  public ByteBuffer createRootRexNodeByteBuffer(final Integer dataOffset) {
    RexNode.startRexNode(flatBufferBuilder);
    RexNode.addData(flatBufferBuilder, dataOffset);
    RexNode.addType(flatBufferBuilder, RexNodeType.Root);
    final Integer rootRexNodeOffset = RexNode.endRexNode(flatBufferBuilder);
    flatBufferBuilder.finish(rootRexNodeOffset);
    return flatBufferBuilder.dataBuffer();
  }

  public Integer createRexCallNodeOffset(final SqlKind sqlKind,
                                         final SqlTypeName sqlTypeName,
                                         final Collection<Integer> operands) {
    Validate.notEmpty(operands, "Call operands are required");
    return createRexNodeOffset(RexNodeType.Call,
                               (short) sqlKind.ordinal(),
                               (short) sqlTypeName.ordinal(),
                               0);
  }

  public Integer createRexCallNodeOffset(final SqlKind sqlKind,
                                         final SqlTypeName sqlTypeName,
                                         final int... operandOffsets) {
    // Validate.notEmpty(operandOffsets, "Call operands are required");
    final Integer operandsOffset =
        RexCall.createOperandsVector(flatBufferBuilder, operandOffsets);
    return createRexNodeOffset(RexNodeType.Call,
                               (short) sqlKind.ordinal(),
                               (short) sqlTypeName.ordinal(),
                               operandsOffset);
  }

  public Integer createRexLiteralNodeOffset(final SqlTypeName sqlTypeName,
                                            final             byte[] value) {
    FlatBufferBuilder localFlatBufferBuilder = new FlatBufferBuilder(0);
    final Integer valueOffset =
        RexLiteral.createValueVector(localFlatBufferBuilder, value);
    final Integer rexLiteralOffset =
        RexLiteral.createRexLiteral(localFlatBufferBuilder, valueOffset);
    localFlatBufferBuilder.finish(rexLiteralOffset);
    final byte[] rexLiteralBytes = localFlatBufferBuilder.sizedByteArray();
    localFlatBufferBuilder       = null;
    return createRexNodeOffset(
        RexNodeType.Literal, SqlKind.LITERAL, sqlTypeName, rexLiteralBytes);
  }

  public Integer
  createRexInputRefNodeOffset(final Collection<Integer> indices) {
    FlatBufferBuilder localFlatBufferBuilder = new FlatBufferBuilder(0);
    int[] indexData = indices.stream().mapToInt(Integer::intValue).toArray();
    final Integer indexOffset =
        RexInputRef.createIndexVector(localFlatBufferBuilder, indexData);
    final Integer rexInputRefOffset =
        RexInputRef.createRexInputRef(localFlatBufferBuilder, indexOffset);
    localFlatBufferBuilder.finish(rexInputRefOffset);
    final byte[] rexInputRefBytes = localFlatBufferBuilder.sizedByteArray();
    localFlatBufferBuilder        = null;
    return createRexNodeOffset(
        RexNodeType.InputRef, SqlKind.INPUT_REF, rexInputRefBytes);
  }

  protected Integer createRexNodeOffset(final Short rexNodeType,
                                        final SqlKind sqlKind,
                                        final SqlTypeName sqlTypeName,
                                        final             byte[] data) {
    return createRexNodeOffset(rexNodeType,
                               (short) sqlKind.ordinal(),
                               (short) sqlTypeName.ordinal(),
                               data);
  }

  protected Integer createRexNodeOffset(final Short rexNodeType,
                                        final Short sqlKind,
                                        final Short sqlTypeName,
                                        final       byte[] data) {
    return createRexNodeOffset(
        rexNodeType, sqlKind, Optional.of(sqlTypeName), data);
  }

  protected Integer createRexNodeOffset(final Short rexNodeType,
                                        final SqlKind sqlKind,
                                        final         byte[] data) {
    return createRexNodeOffset(
        rexNodeType, (short) sqlKind.ordinal(), Optional.empty(), data);
  }

  protected Integer createRexNodeOffset(final Short rexNodeType,
                                        final Short sqlKind,
                                        final       byte[] data) {
    return createRexNodeOffset(rexNodeType, sqlKind, Optional.empty(), data);
  }

  protected Integer createRexNodeOffset(final Short rexNodeType,
                                        final Short sqlKind,
                                        final Short sqlTypeName,
                                        final Integer dataOffset) {
    return createRexNodeOffset(
        rexNodeType, sqlKind, Optional.of(sqlTypeName), dataOffset);
  }

  protected Integer createRexNodeOffset(final Short rexNodeType,
                                        final Short sqlKind,
                                        final Integer dataOffset) {
    return createRexNodeOffset(
        rexNodeType, sqlKind, Optional.empty(), dataOffset);
  }

  protected Integer
  createRexNodeOffset(final Short rexNodeType,
                      final Short sqlKind,
                      final Optional<Short> sqlTypeNameOptional,
                      final                 byte[] data) {
    final Integer dataOffset =
        RexNode.createDataVector(flatBufferBuilder, data);
    return createRexNodeOffset(
        rexNodeType, sqlKind, sqlTypeNameOptional, dataOffset);
  }

  protected Integer
  createRexNodeOffset(final Short rexNodeType,
                      final Short sqlKind,
                      final Optional<Short> sqlTypeNameOptional,
                      final Integer dataOffset) {
    return RexNode.createRexNode(flatBufferBuilder,
                                 rexNodeType,
                                 sqlKind,
                                 sqlTypeNameOptional.orElse((short) -1),
                                 dataOffset);
  }
}
