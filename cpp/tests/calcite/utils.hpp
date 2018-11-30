#include "../../src/blazingdb/protocol/message/generated/all_generated.h"

namespace factory {
using namespace com::blazingdb::protocol::calcite::plan::messages;

flatbuffers::DetachedBuffer
CreateTableScanDetachedBuffer(const std::vector<std::string> &qualifiedName);

flatbuffers::DetachedBuffer CreateLogicalUnionDetachedBuffer(const bool all);

}  // namespace factory

