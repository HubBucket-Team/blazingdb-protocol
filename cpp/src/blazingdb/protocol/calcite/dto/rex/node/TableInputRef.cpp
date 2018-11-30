#include "rex/node/TableInputRef.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

TableInputRef::TableInputRef(KindName kind, TypeName type)
 : RexCall(kind, type)
{ }

void TableInputRef::accept(RexVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
