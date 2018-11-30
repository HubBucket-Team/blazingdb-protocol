#include "rex/node/LocalRef.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

LocalRef::LocalRef(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void LocalRef::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
