#include "rex/node/DynamicParam.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

DynamicParam::DynamicParam(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void DynamicParam::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
