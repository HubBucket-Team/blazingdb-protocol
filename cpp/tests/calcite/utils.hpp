#include "../../src/blazingdb/protocol/message/generated/all_generated.h"

namespace factory {
using namespace com::blazingdb::protocol::calcite::plan::messages;

flatbuffers::DetachedBuffer CreateTableScanNodeDetachedBuffer(
    const std::vector<std::string> &qualifiedName);

flatbuffers::DetachedBuffer CreateLogicalUnionNodeDetachedBuffer(
    const bool                         all,
    const flatbuffers::DetachedBuffer &leftNodeDetachedBuffer,
    const flatbuffers::DetachedBuffer &rightNodeDetachedBuffer);

flatbuffers::DetachedBuffer CreateLogicalProjectNodeDetachedBuffer(
    const std::vector<std::string> &   columnNames,
    const std::vector<std::size_t> &   columnIndices,
    const flatbuffers::DetachedBuffer &tableScanNodeDetachedBuffer);

}  // namespace factory

