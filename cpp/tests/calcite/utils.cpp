#include "utils.hpp"
#include <iostream>

using namespace com::blazingdb::protocol::calcite::plan::messages;

namespace {

flatbuffers::Offset<RelNode>
CreateInputOffset(flatbuffers::FlatBufferBuilder &   flatBufferBuilder,
                  const flatbuffers::DetachedBuffer &detachedBuffer) {
    const std::size_t size = detachedBuffer.size();
    flatBufferBuilder.StartVector(size, 4);
    flatBufferBuilder.PushBytes(detachedBuffer.data(), size);
    return flatBufferBuilder.EndVector(size);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<RelNode>>>
CreateInputsVector(flatbuffers::FlatBufferBuilder &flatBufferBuilder,
                   const std::initializer_list<flatbuffers::DetachedBuffer *>
                       &inputDetachedBuffers) {
    auto inputSize           = inputDetachedBuffers.size();
    auto inputDetachedBuffer = inputDetachedBuffers.begin();

    flatBufferBuilder.StartVector(inputSize,
                                  sizeof(flatbuffers::Offset<RelNode>));
    for (auto i = inputSize; i > 0;) {
        flatBufferBuilder.PushElement(inputDetachedBuffer[--i]);
    }
    auto inputsOffset =
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<RelNode>>>(
            flatBufferBuilder.EndVector(inputSize));

    return inputsOffset;
}

flatbuffers::DetachedBuffer CreateRelNodeDetachedBuffer(
    flatbuffers::FlatBufferBuilder &   flatBufferBuilder,
    const RelNodeType                  relNodeType,
    const flatbuffers::DetachedBuffer &dataDetachedBuffer) {
    auto dataOffset = flatBufferBuilder.CreateVector(
        reinterpret_cast<std::int8_t *>(
            const_cast<std::uint8_t *>(dataDetachedBuffer.data())),
        dataDetachedBuffer.size());
    auto relNodeOffset =
        CreateRelNode(flatBufferBuilder, relNodeType, dataOffset);
    flatBufferBuilder.Finish(relNodeOffset);
    return flatBufferBuilder.Release();
}

flatbuffers::DetachedBuffer CreateRelNodeDetachedBuffer(
    const RelNodeType                  relNodeType,
    const flatbuffers::DetachedBuffer &dataDetachedBuffer) {
    flatbuffers::FlatBufferBuilder flatBufferBuilder(0);
    auto                           dataOffset = flatBufferBuilder.CreateVector(
        reinterpret_cast<std::int8_t *>(
            const_cast<std::uint8_t *>(dataDetachedBuffer.data())),
        dataDetachedBuffer.size());
    auto relNodeOffset =
        CreateRelNode(flatBufferBuilder, relNodeType, dataOffset);
    flatBufferBuilder.Finish(relNodeOffset);
    return flatBufferBuilder.Release();
}

template <class Input, class... Inputs>
void AddInputs(flatbuffers::FlatBufferBuilder &flatBufferBuilder,
               flatbuffers::Offset<RelNode> *  inputOffsets,
               Input &                         inputDetachedBuffer) {
    auto input = flatbuffers::GetRoot<RelNode>(inputDetachedBuffer.data());
    std::vector<std::int8_t> data(input->data()->begin(), input->data()->end());
    flatbuffers::Offset<RelNode> inputOffset =
        CreateRelNodeDirect(flatBufferBuilder, input->type(), &data);
    std::memcpy(
        inputOffsets, &inputOffset, sizeof(flatbuffers::Offset<RelNode>));
}

template <class Input, class... Inputs>
void AddInputs(flatbuffers::FlatBufferBuilder &flatBufferBuilder,
               flatbuffers::Offset<RelNode> *  inputOffsets,
               Input &                         inputDetachedBuffer,
               Inputs &... inputDetachedBuffers) {
    AddInputs(flatBufferBuilder, inputOffsets, inputDetachedBuffer);
    AddInputs(flatBufferBuilder, ++inputOffsets, inputDetachedBuffers...);
}

template <class... Inputs>
flatbuffers::DetachedBuffer CreateRelNodeDetachedBuffer(
    const RelNodeType                  relNodeType,
    const flatbuffers::DetachedBuffer &dataDetachedBuffer,
    Inputs &... inputs) {
    flatbuffers::FlatBufferBuilder flatBufferBuilder(0);

    auto dataOffset = flatBufferBuilder.CreateVector(
        reinterpret_cast<std::int8_t *>(
            const_cast<std::uint8_t *>(dataDetachedBuffer.data())),
        dataDetachedBuffer.size());

    flatbuffers::Offset<RelNode> inputOffsets[sizeof...(inputs)];
    AddInputs(flatBufferBuilder, inputOffsets, inputs...);
    auto inputsOffset =
        flatBufferBuilder.CreateVector(inputOffsets, sizeof...(inputs));

    auto relNodeOffset =
        CreateRelNode(flatBufferBuilder, relNodeType, dataOffset, inputsOffset);
    flatBufferBuilder.Finish(relNodeOffset);

    return flatBufferBuilder.Release();
}

}  // namespace

namespace factory {

flatbuffers::DetachedBuffer CreateTableScanNodeDetachedBuffer(
    const std::vector<std::string> &qualifiedName) {
    flatbuffers::FlatBufferBuilder flatBufferBuilder(0);
    auto                           qualifiedNameOffset =
        flatBufferBuilder.CreateVectorOfStrings(qualifiedName);
    auto tableScanOffset =
        CreateTableScan(flatBufferBuilder, qualifiedNameOffset);
    flatBufferBuilder.Finish(tableScanOffset);
    auto tableScanDetachedBuffer = flatBufferBuilder.Release();
    return CreateRelNodeDetachedBuffer(RelNodeType_TableScan,
                                       tableScanDetachedBuffer);
}

flatbuffers::DetachedBuffer CreateLogicalUnionNodeDetachedBuffer(
    const bool                         all,
    const flatbuffers::DetachedBuffer &leftNodeDetachedBuffer,
    const flatbuffers::DetachedBuffer &rightNodeDetachedBuffer) {
    flatbuffers::FlatBufferBuilder flatBufferBuilder(0);
    auto logicalUnionOffset = CreateLogicalUnion(flatBufferBuilder, all);
    flatBufferBuilder.Finish(logicalUnionOffset);
    auto logicalUnionDetachedBuffer = flatBufferBuilder.Release();
    return CreateRelNodeDetachedBuffer(RelNodeType_LogicalUnion,
                                       logicalUnionDetachedBuffer,
                                       leftNodeDetachedBuffer,
                                       rightNodeDetachedBuffer);
}

flatbuffers::DetachedBuffer CreateLogicalProjectNodeDetachedBuffer(
    const std::vector<std::string> &   columnNames,
    const std::vector<std::size_t> &   columnIndices,
    const flatbuffers::DetachedBuffer &tableScanNodeDetachedBuffer) {
    flatbuffers::FlatBufferBuilder flatBufferBuilder(0);
    auto                           columnNamesOffset =
        flatBufferBuilder.CreateVectorOfStrings(columnNames);
    auto columnIndicesOffset  = flatBufferBuilder.CreateVector(columnIndices);
    auto logicalProjectOffset = CreateLogicalProject(
        flatBufferBuilder, columnNamesOffset, columnIndicesOffset);
    flatBufferBuilder.Finish(logicalProjectOffset);
    auto logicalProjectDetachedBuffer = flatBufferBuilder.Release();
    return CreateRelNodeDetachedBuffer(RelNodeType_LogicalProject,
                                       logicalProjectDetachedBuffer,
                                       tableScanNodeDetachedBuffer);
}

}  // namespace factory
