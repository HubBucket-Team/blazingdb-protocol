#include "rex/node/RangeRef.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

RangeRef::RangeRef(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void RangeRef::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
