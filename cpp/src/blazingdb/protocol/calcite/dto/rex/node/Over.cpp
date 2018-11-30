#include "rex/node/Over.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

Over::Over(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void Over::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
