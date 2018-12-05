#include "rel/node/LogicalUnion.h"
#include "rel/visitor/RelVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

LogicalUnion::LogicalUnion(bool all)
 : all {all}
{ }

void LogicalUnion::accept(RelVisitor* visitor) {
    visitor->visit(this);
}

bool LogicalUnion::getAll() const {
    return all;
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
