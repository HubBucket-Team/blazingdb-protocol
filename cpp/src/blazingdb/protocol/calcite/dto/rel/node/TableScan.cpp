#include "rel/node/TableScan.h"
#include "rel/visitor/RelVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

void TableScan::accept(RelVisitor* visitor) {
    visitor->visit(this);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
