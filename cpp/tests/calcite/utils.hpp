#include "../../src/blazingdb/protocol/message/generated/all_generated.h"

namespace factory {
using namespace com::blazingdb::protocol::calcite::plan::messages;

flatbuffers::DetachedBuffer CreateTableScanNodeDetachedBuffer(
    const std::vector<std::string> &qualifiedName);

flatbuffers::DetachedBuffer CreateLogicalUnionNodeDetachedBuffer(
    const bool                         all,
    const flatbuffers::DetachedBuffer &leftNodeDetachedBuffer,
    const flatbuffers::DetachedBuffer &rightNodeDetachedBuffer);

}  // namespace factory

