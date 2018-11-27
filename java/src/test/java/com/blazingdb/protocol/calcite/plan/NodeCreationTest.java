package com.blazingdb.protocol.calcite.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import com.blazingdb.protocol.calcite.plan.RelNodeFactory;
import com.blazingdb.protocol.calcite.plan.messages.RelNode;
import com.blazingdb.protocol.calcite.plan.messages.RelNodeType;
import com.blazingdb.protocol.calcite.plan.messages.TableScan;

import org.junit.Test;

import java.nio.ByteBuffer;

import java.util.Arrays;
import java.util.List;

public final class NodeCreationTest {

  @Test
  public void testCreation() {
    RelNodeFactory relNodeFactory = new RelNodeFactory();

    final List<String> leftQualifiedName  = Arrays.asList("left", "table");
    final List<String> rightQualifiedName = Arrays.asList("right", "table");

    final Integer leftTableScanRelNodeOffset =
        relNodeFactory.createTableScanNodeOffset();

    final Integer rootRelNodeOffset =
        relNodeFactory.createRootRelNodeOffset(leftTableScanRelNodeOffset);

    relNodeFactory.finish(rootRelNodeOffset);
    final byte[] rootRelNodeBytes = relNodeFactory.makeDataBuffer();
    final RelNode rootRelNode =
        RelNode.getRootAsRelNode(ByteBuffer.wrap(rootRelNodeBytes));

    assertEquals(RelNodeType.Root, rootRelNode.type());
    assertEquals(0, rootRelNode.dataLength());
    assertEquals(1, rootRelNode.inputsLength());

    final RelNode leftTableScanRelNode = rootRelNode.inputs(0);
    assertNotEquals(0, leftTableScanRelNode.dataLength());
    assertEquals(RelNodeType.TableScan, leftTableScanRelNode.type());
    assertEquals(0, leftTableScanRelNode.inputsLength());

    final byte[] leftTableScanBytes =
        leftTableScanRelNode.dataAsByteBuffer().array();
    final TableScan leftTableScan =
        TableScan.getRootAsTableScan(ByteBuffer.wrap(leftTableScanBytes));
    assertEquals(2, leftTableScan.qualifiedNameLength());
  }
}
