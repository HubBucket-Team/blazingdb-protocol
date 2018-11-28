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
  }

  @Test
  public void testSimpleNestedRexCall() {}
}
