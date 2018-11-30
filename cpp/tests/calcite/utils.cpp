#include "utils.hpp"

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
    std::vector<flatbuffers::DetachedBuffer *> reversed{inputDetachedBuffers};

    std::vector<flatbuffers::Offset<RelNode>> results(reversed.size());

    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it) {
        auto &db     = *it;
        auto  offset = CreateInputOffset(flatBufferBuilder, *db);
        results.push_back(offset);
    }

    auto inputsOffset = flatBufferBuilder.CreateVector(results);

    return inputsOffset;
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

}  // namespace

namespace factory {

flatbuffers::DetachedBuffer
CreateTableScanDetachedBuffer(const std::vector<std::string> &qualifiedName) {
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

flatbuffers::DetachedBuffer CreateLogicalUnionDetachedBuffer(const bool all) {
    flatbuffers::FlatBufferBuilder flatBufferBuilder(0);
    auto logicalUnionOffset = CreateLogicalUnion(flatBufferBuilder, all);
    flatBufferBuilder.Finish(logicalUnionOffset);
    return flatBufferBuilder.Release();
}

}  // namespace factory
