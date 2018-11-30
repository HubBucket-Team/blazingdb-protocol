#include "rex/node/Call.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

Call::Call(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void Call::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
