package com.blazingdb.protocol.calcite.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import com.blazingdb.protocol.calcite.plan.RelNodeFactory;
import com.blazingdb.protocol.calcite.plan.messages.LogicalUnion;
import com.blazingdb.protocol.calcite.plan.messages.RelNode;
import com.blazingdb.protocol.calcite.plan.messages.RelNodeType;
import com.blazingdb.protocol.calcite.plan.messages.TableScan;

import org.junit.Test;

import java.nio.ByteBuffer;

import java.util.Arrays;
import java.util.List;

public final class NodeCreationTest {

  @Test
  public void testSimpleNested() {
    RelNodeFactory relNodeFactory = new RelNodeFactory();

    final List<String> leftQualifiedName  = Arrays.asList("left", "table");
    final List<String> rightQualifiedName = Arrays.asList("right", "table");

    final Integer leftTableScanRelNodeOffset =
        relNodeFactory.createTableScanRelNodeOffset(leftQualifiedName);
    final Integer rightTableScanRelNodeOffset =
        relNodeFactory.createTableScanRelNodeOffset(rightQualifiedName);

    final Integer unionRelNodeOffset =
        relNodeFactory.createLogicalUnionRelNodeOffset(
            true, leftTableScanRelNodeOffset, rightTableScanRelNodeOffset);

    final ByteBuffer rootRelNodeByteBuffer =
        relNodeFactory.createRootRelNodeOffset(unionRelNodeOffset);

    final RelNode rootRelNode = RelNode.getRootAsRelNode(rootRelNodeByteBuffer);

    assertEquals(RelNodeType.Root, rootRelNode.type());
    assertEquals(0, rootRelNode.dataLength());
    assertEquals(1, rootRelNode.inputsLength());

    final RelNode logicalUnionRelNode = rootRelNode.inputs(0);
    assertEquals(RelNodeType.LogicalUnion, logicalUnionRelNode.type());
    assertNotEquals(0, logicalUnionRelNode.dataLength());
    assertEquals(2, logicalUnionRelNode.inputsLength());

    final LogicalUnion logicalUnion = LogicalUnion.getRootAsLogicalUnion(
        logicalUnionRelNode.dataAsByteBuffer());
    assertTrue(logicalUnion.all());

    final RelNode leftTableScanRelNode = logicalUnionRelNode.inputs(0);
    assertNotEquals(0, leftTableScanRelNode.dataLength());
    assertEquals(RelNodeType.TableScan, leftTableScanRelNode.type());
    assertEquals(0, leftTableScanRelNode.inputsLength());

    final TableScan leftTableScan =
        TableScan.getRootAsTableScan(leftTableScanRelNode.dataAsByteBuffer());
    assertEquals(2, leftTableScan.qualifiedNameLength());
    assertEquals("left", leftTableScan.qualifiedName(0));
    assertEquals("table", leftTableScan.qualifiedName(1));

    final RelNode rightTableScanRelNode = logicalUnionRelNode.inputs(1);
    assertNotEquals(0, rightTableScanRelNode.dataLength());
    assertEquals(RelNodeType.TableScan, rightTableScanRelNode.type());
    assertEquals(0, rightTableScanRelNode.inputsLength());

    final TableScan rightTableScan =
        TableScan.getRootAsTableScan(rightTableScanRelNode.dataAsByteBuffer());
    assertEquals(2, rightTableScan.qualifiedNameLength());
    assertEquals("right", rightTableScan.qualifiedName(0));
    assertEquals("table", rightTableScan.qualifiedName(1));
  }
}
