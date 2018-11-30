#include "rex/node/Literal.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

Literal::Literal(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void Literal::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
