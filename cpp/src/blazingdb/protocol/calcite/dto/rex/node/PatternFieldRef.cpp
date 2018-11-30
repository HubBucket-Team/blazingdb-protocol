#include "rex/node/PatternFieldRef.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

PatternFieldRef::PatternFieldRef(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void PatternFieldRef::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
