package com.blazingdb.protocol.calcite.plan;

import com.blazingdb.protocol.calcite.plan.messages.Node;
import com.blazingdb.protocol.calcite.plan.messages.NodeType;

import com.google.flatbuffers.FlatBufferBuilder;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.Validate;

import java.util.Collection;

import java.nio.ByteBuffer;

final class NodeFactory {

  private final FlatBufferBuilder flatBufferBuilder;

  // TODO(gcca): test `ContextFBPool` (for memory management on building)
  public NodeFactory(final FlatBufferBuilder flatBufferBuilder) {
    Validate.notNull(flatBufferBuilder, "FlatBufferBuilder is required");
    this.flatBufferBuilder = flatBufferBuilder;
  }

  // TODO(gcca): abstract node factory
  public NodeFactory() { this.flatBufferBuilder = new FlatBufferBuilder(0); }

  public NodeFactory(int initialSize) {
    this.flatBufferBuilder = new FlatBufferBuilder(initialSize);
  }

  public Node createNode(final Collection<Node> children) {
    throw new NotImplementedException("Build from messages");
  }

  public int createNode(final byte[] data, final int[] childOffsets) {
    int dataOffset = Node.createDataVector(flatBufferBuilder, data);
    int childrenOffset =
        Node.createChildrenVector(flatBufferBuilder, childOffsets);
    return Node.createNode(flatBufferBuilder, NodeType.Root, dataOffset,
                           childrenOffset);
  }

  public ByteBuffer getDataBuffer() { return flatBufferBuilder.dataBuffer(); }

  public byte[] makeDataBuffer() { return flatBufferBuilder.sizedByteArray(); }
}
