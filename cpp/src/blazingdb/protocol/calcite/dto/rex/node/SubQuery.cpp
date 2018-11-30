#include "rex/node/SubQuery.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

SubQuery::SubQuery(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void SubQuery::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
