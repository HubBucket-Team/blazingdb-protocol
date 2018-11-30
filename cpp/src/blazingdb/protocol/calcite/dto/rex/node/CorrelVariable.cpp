#include "rex/node/CorrelVariable.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

CorrelVariable::CorrelVariable(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void CorrelVariable::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
