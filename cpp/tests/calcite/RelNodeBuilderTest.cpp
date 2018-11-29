#include "../../src/blazingdb/protocol/message/generated/all_generated.h"
#include <blazingdb/protocol/calcite/messages/RelNodeBuilder.hpp>
#include <gtest/gtest.h>

namespace factory {
using namespace com::blazingdb::protocol::calcite::plan::messages;

flatbuffers::DetachedBuffer
createTableScanOffset(const std::vector<std::string> &qualifiedName);

flatbuffers::DetachedBuffer
createTableScanOffset(const std::vector<std::string> &qualifiedName) {
    flatbuffers::FlatBufferBuilder fbb(0);
    auto qualifiedNameOffset = fbb.CreateVectorOfStrings(qualifiedName);
    auto tableScanOffset     = CreateTableScan(fbb, qualifiedNameOffset);
    fbb.Finish(tableScanOffset);
    return fbb.Release();
}

}  // namespace factory

void CreateTree();

void CreateTree() {
    using namespace com::blazingdb::protocol::calcite::plan::messages;

    flatbuffers::FlatBufferBuilder fbb(0);

    auto rootNodeOffset =
        CreateRelNode(fbb, RelNodeType::RelNodeType_Root, 0, 0);
}

TEST(RelNodeBuilderTest, Main) {
    using namespace blazingdb::protocol::calcite::messages;

    CreateTree();

    const std::size_t DATA_SIZE = 512;
    std::uint8_t      data[DATA_SIZE];

    RelNodeBuilder relNodeBilder(data);
    relNodeBilder.Build();
}
