package com.blazingdb.protocol.calcite.plan;

import com.blazingdb.protocol.calcite.plan.messages.RelNode;
import com.blazingdb.protocol.calcite.plan.messages.RelNodeType;
import com.blazingdb.protocol.calcite.plan.messages.TableScan;

import com.google.flatbuffers.FlatBufferBuilder;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.Validate;

import java.nio.ByteBuffer;

import java.util.Collection;

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

  public Integer createTableScanNodeOffset() {
    FlatBufferBuilder localFlatBufferBuilder = new FlatBufferBuilder(0);

    int[] ss = new int[2];
    ss[0]    = localFlatBufferBuilder.createString("LOCAL");
    ss[1]    = localFlatBufferBuilder.createString("THREAD");

    final Integer qualifiedNameOffset =
        TableScan.createQualifiedNameVector(localFlatBufferBuilder, ss);

    final Integer tableScanOffset =
        TableScan.createTableScan(localFlatBufferBuilder, qualifiedNameOffset);

    localFlatBufferBuilder.finish(tableScanOffset);
    final byte[] tableScanBytes = localFlatBufferBuilder.sizedByteArray();
    localFlatBufferBuilder      = null;

    return createRelNodeOffset(RelNodeType.TableScan, tableScanBytes);
  }

  protected Integer createRelNodeOffset(final short relNodeType,
                                        final       byte[] data,
                                        final int... inputOffsets) {
    final FlatBufferBuilder localFlatBufferBuilder = new FlatBufferBuilder(0);
    final int               dataOffset =
        RelNode.createDataVector(localFlatBufferBuilder, data);
    return createRelNodeOffset(
        localFlatBufferBuilder, relNodeType, dataOffset, inputOffsets);
  }

  protected Integer
  createRelNodeOffset(final FlatBufferBuilder localFlatBufferBuilder,
                      final short             relNodeType,
                      final Integer dataOffset,
                      final int... inputOffsets) {
    Integer relNodeOffset;

    if (0 == inputOffsets.length) {
      RelNode.startRelNode(localFlatBufferBuilder);
      RelNode.addData(localFlatBufferBuilder, dataOffset);
      RelNode.addType(localFlatBufferBuilder, relNodeType);
      relNodeOffset = RelNode.endRelNode(localFlatBufferBuilder);
    } else {
      final Integer inputsOffset =
          RelNode.createInputsVector(localFlatBufferBuilder, inputOffsets);
      relNodeOffset = RelNode.createRelNode(
          localFlatBufferBuilder, relNodeType, dataOffset, inputsOffset);
    }

    localFlatBufferBuilder.finish(relNodeOffset);
    final byte[] relNodeBytes = localFlatBufferBuilder.sizedByteArray();

    return flatBufferBuilder.createByteVector(relNodeBytes);
  }
}
