package com.blazingdb.protocol.calcite.plan;

import com.blazingdb.protocol.calcite.plan.messages.Node;
import com.blazingdb.protocol.calcite.plan.messages.NodeType;
import com.blazingdb.protocol.calcite.plan.messages.TableScanData;
import com.blazingdb.protocol.calcite.plan.messages.UnionData;

import com.google.flatbuffers.FlatBufferBuilder;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.Validate;

import java.util.Collection;
import java.util.List;

import java.nio.ByteBuffer;

final class NodeFactory {

  private final FlatBufferBuilder flatBufferBuilder;

  // TODO(gcca): test `ContextFBObjectsPool` (on building)
  public NodeFactory(final FlatBufferBuilder flatBufferBuilder) {
    Validate.notNull(flatBufferBuilder, "FlatBufferBuilder is required");
    this.flatBufferBuilder = flatBufferBuilder;
  }

  // TODO(gcca): abstract node factory
  public NodeFactory() { this.flatBufferBuilder = new FlatBufferBuilder(0); }

  public NodeFactory(Integer initialSize) {
    this.flatBufferBuilder = new FlatBufferBuilder(initialSize);
  }

  public Node createNode(final Collection<Node> children) {
    throw new NotImplementedException("Build from messages");
  }

  public ByteBuffer getDataBuffer() { return flatBufferBuilder.dataBuffer(); }

  public byte[] makeDataBuffer() { return flatBufferBuilder.sizedByteArray(); }

  public Integer createTableScanOffset(final List<String> qualifiedName) {
    final Integer qualifiedNameOffset = TableScanData.createQualifiedNameVector(
        flatBufferBuilder,
        qualifiedName.stream()
            .mapToInt(flatBufferBuilder::createString)
            .toArray());
    final Integer tableScanDataOffset = TableScanData.createTableScanData(
        flatBufferBuilder, qualifiedNameOffset);
    return createNodeOffsetWithoutChildren(NodeType.TableScan,
                                           tableScanDataOffset);
  }

  public Integer createUnionOffset(final boolean all,
                                   final Integer leftOffset,
                                   final Integer rightOffset) {
    final Integer unionDataOffset =
        UnionData.createUnionData(flatBufferBuilder, all);
    return createNodeOffsetWithoutChildren(NodeType.Union, unionDataOffset);
  }

  protected Integer createNodeOffsetWithoutChildren(final byte nodeType,
                                                    final Integer dataOffset) {
    return Node.createNode(flatBufferBuilder, nodeType, dataOffset, 0);
  }
}
