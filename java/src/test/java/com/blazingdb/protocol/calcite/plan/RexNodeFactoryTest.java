package com.blazingdb.protocol.calcite.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import com.blazingdb.protocol.calcite.plan.messages.RexCall;
import com.blazingdb.protocol.calcite.plan.messages.RexInputRef;
import com.blazingdb.protocol.calcite.plan.messages.RexLiteral;
import com.blazingdb.protocol.calcite.plan.messages.RexNode;
import com.blazingdb.protocol.calcite.plan.messages.RexNodeType;

import org.junit.Test;

import java.nio.ByteBuffer;

import java.util.Arrays;
import java.util.List;

public final class RexNodeFactoryTest {

  @Test
  public void testSimpleRexCall() {
    RexNodeFactory rexNodeFactory = new RexNodeFactory();

    final List<Integer> indices = Arrays.asList(1, 2, 3);
    final Integer rexInputRefNodeOffset =
        rexNodeFactory.createRexInputRefNodeOffset(indices);

    final ByteBuffer rexLiteralValueByteBuffer =
        ByteBuffer.allocate(Integer.BYTES);
    rexLiteralValueByteBuffer.putInt(123);
    final Integer rexLiteralNodeOffset =
        rexNodeFactory.createRexLiteralNodeOffset(SqlTypeName.INTEGER,
                                                  rexLiteralValueByteBuffer);

    final Integer rexCallNodeOffset =
        rexNodeFactory.createRexCallNodeOffset(SqlKind.EQUALS,
                                               SqlTypeName.BOOLEAN,
                                               rexInputRefNodeOffset,
                                               rexLiteralNodeOffset);

    rexNodeFactory.flatBufferBuilder.finish(rexCallNodeOffset);
    final ByteBuffer rexCallByteBuffer =
        rexNodeFactory.flatBufferBuilder.dataBuffer();

    final RexNode rexCallNode = RexNode.getRootAsRexNode(rexCallByteBuffer);
    assertEquals(RexNodeType.Call, rexCallNode.type());
    assertEquals(SqlKind.EQUALS.ordinal(), rexCallNode.sqlKind());
    assertEquals(SqlTypeName.BOOLEAN.ordinal(), rexCallNode.sqlTypeName());
    assertNotEquals(0, rexCallNode.dataLength());

    final RexCall rexCall =
        RexCall.getRootAsRexCall(rexCallNode.dataAsByteBuffer());
    assertEquals(2, rexCall.operandsLength());

    final RexNode rexInputRefNode = rexCall.operands(0);
    assertEquals(RexNodeType.InputRef, rexInputRefNode.type());
    assertEquals(SqlKind.INPUT_REF.ordinal(), rexInputRefNode.sqlKind());
    assertEquals(-1, rexInputRefNode.sqlTypeName());

    final RexInputRef rexInputRef =
        RexInputRef.getRootAsRexInputRef(rexInputRefNode.dataAsByteBuffer());
    assertEquals(3, rexInputRef.indexLength());
    assertEquals(1, rexInputRef.index(0));
    assertEquals(2, rexInputRef.index(1));
    assertEquals(3, rexInputRef.index(2));

    final RexNode rexLiteralNode = rexCall.operands(1);
    assertEquals(RexNodeType.Literal, rexLiteralNode.type());
    assertEquals(SqlKind.LITERAL.ordinal(), rexLiteralNode.sqlKind());
    assertEquals(SqlTypeName.INTEGER.ordinal(), rexLiteralNode.sqlTypeName());

    final RexLiteral rexLiteral =
        RexLiteral.getRootAsRexLiteral(rexLiteralNode.dataAsByteBuffer());
    assertEquals(4, rexLiteral.valueLength());

    final byte[] valueBytes = new byte[4];
    final ByteBuffer valueByteBuffer = rexLiteral.valueAsByteBuffer();
    valueByteBuffer.get(valueBytes);
    assertEquals(123, ByteBuffer.wrap(valueBytes).getInt());
  }

  @Test
  public void testSimpleNestedRexCall() {}
}
