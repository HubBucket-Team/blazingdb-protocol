package com.blazingdb.protocol.calcite.plan;

import com.blazingdb.protocol.calcite.plan.messages.LogicalUnion;
import com.blazingdb.protocol.calcite.plan.messages.RelNode;
import com.blazingdb.protocol.calcite.plan.messages.RelNodeType;
import com.blazingdb.protocol.calcite.plan.messages.TableScan;

import com.google.flatbuffers.FlatBufferBuilder;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.Validate;

import java.nio.ByteBuffer;

import java.util.Collection;
import java.util.List;

final class RelNodeFactory {

  private final FlatBufferBuilder flatBufferBuilder;

  // TODO(gcca): test `ContextFBObjectsPool` (on building)
  public RelNodeFactory(final FlatBufferBuilder flatBufferBuilder) {
    Validate.notNull(flatBufferBuilder, "FlatBufferBuilder is required");
    this.flatBufferBuilder = flatBufferBuilder;
  }

  // TODO(gcca): abstract node factory
  public RelNodeFactory(Integer initialSize) {
    this(new FlatBufferBuilder(initialSize));
  }

  // TODO(gcca): idem
  public RelNodeFactory() { this(0); }

  public RelNode createNode(final Collection<RelNode> children) {
    throw new NotImplementedException("Build from messages");
  }

  public void finish(final Integer rootNodeOffset) {
    flatBufferBuilder.finish(rootNodeOffset);
  }

  public ByteBuffer getDataBuffer() { return flatBufferBuilder.dataBuffer(); }

  public byte[] makeDataBuffer() { return flatBufferBuilder.sizedByteArray(); }

  public Integer createRootRelNodeOffset(final int... inputOffsets) {
    final Integer inputsOffset =
        RelNode.createInputsVector(flatBufferBuilder, inputOffsets);
    RelNode.startRelNode(flatBufferBuilder);
    RelNode.addInputs(flatBufferBuilder, inputsOffset);
    RelNode.addType(flatBufferBuilder, RelNodeType.Root);
    return RelNode.endRelNode(flatBufferBuilder);
  }

  public Integer
  createTableScanRelNodeOffset(final List<String> qualifiedName) {
    FlatBufferBuilder localFlatBufferBuilder = new FlatBufferBuilder(0);
    final int[] qualifiedNameData =
        qualifiedName.stream()
            .mapToInt(localFlatBufferBuilder::createString)
            .toArray();

    final Integer qualifiedNameOffset = TableScan.createQualifiedNameVector(
        localFlatBufferBuilder, qualifiedNameData);

    final Integer tableScanOffset =
        TableScan.createTableScan(localFlatBufferBuilder, qualifiedNameOffset);

    localFlatBufferBuilder.finish(tableScanOffset);
    final byte[] tableScanBytes = localFlatBufferBuilder.sizedByteArray();
    localFlatBufferBuilder      = null;

    return createRelNodeOffset(RelNodeType.TableScan, tableScanBytes);
  }

  public Integer
  createLogicalUnionRelNodeOffset(final Boolean all,
                                  final Integer leftRelNodeOffset,
                                  final Integer rightRelNodeOffset) {
    FlatBufferBuilder localFlatBufferBuilder = new FlatBufferBuilder(0);
    final Integer logicalUnionOffset =
        LogicalUnion.createLogicalUnion(localFlatBufferBuilder, all);
    localFlatBufferBuilder.finish(logicalUnionOffset);
    final byte[] logicalUnionBytes = localFlatBufferBuilder.sizedByteArray();
    localFlatBufferBuilder         = null;
    return createRelNodeOffset(RelNodeType.LogicalUnion,
                               logicalUnionBytes,
                               leftRelNodeOffset,
                               rightRelNodeOffset);
  }

  protected Integer createRelNodeOffset(final short relNodeType,
                                        final       byte[] data,
                                        final int... inputOffsets) {
    final int dataOffset = RelNode.createDataVector(flatBufferBuilder, data);
    return createRelNodeOffset(relNodeType, dataOffset, inputOffsets);
  }

  protected Integer createRelNodeOffset(final short relNodeType,
                                        final Integer dataOffset,
                                        final int... inputOffsets) {
    if (0 == inputOffsets.length) {
      RelNode.startRelNode(flatBufferBuilder);
      RelNode.addData(flatBufferBuilder, dataOffset);
      RelNode.addType(flatBufferBuilder, relNodeType);
      return RelNode.endRelNode(flatBufferBuilder);
    } else {
      final Integer inputsOffset =
          RelNode.createInputsVector(flatBufferBuilder, inputOffsets);
      return RelNode.createRelNode(
          flatBufferBuilder, relNodeType, dataOffset, inputsOffset);
    }
  }
}
