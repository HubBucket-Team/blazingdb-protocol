#include "rex/base/RexCall.h"

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

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
