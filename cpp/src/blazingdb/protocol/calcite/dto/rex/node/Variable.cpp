#include "rex/node/Variable.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

Variable::Variable(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void Variable::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
