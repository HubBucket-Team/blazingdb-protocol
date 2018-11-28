package com.blazingdb.protocol.calcite.plan;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public final class RexNodeFactoryTest {

  @Test
  public void testSimpleNestedRexCall() {
    RexNodeFactory rexNodeFactory = new RexNodeFactory();

    final List<Integer> indices = Arrays.asList(1, 2, 3);

    final Integer rexInputRefNodeOffset =
        rexNodeFactory.createRexInputRefNodeOffset(indices);
  }
}
