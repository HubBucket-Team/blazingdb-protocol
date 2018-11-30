#include "rex/node/InputRef.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

InputRef::InputRef(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void InputRef::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
