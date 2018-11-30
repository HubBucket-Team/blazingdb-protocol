#include "rex/base/RexCall.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

RexCall::RexCall()
{ }

RexCall::RexCall(KindName kind, TypeName type)
 : RexNode(kind, type)
{ }

VectorRexNodePtr& RexCall::getOperands() {
    return operands;
}

void RexCall::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
