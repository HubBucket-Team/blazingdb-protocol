#include "rex/node/FieldAccess.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

FieldAccess::FieldAccess(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void FieldAccess::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
