#include "rel/node/TableScan.h"
#include "rel/visitor/RelVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

TableScan::TableScan(std::vector<std::string>& qualifiedName)
 : qualifiedName{qualifiedName}
{ }

TableScan::TableScan(std::vector<std::string>&& qualifiedName)
 : qualifiedName{std::move(qualifiedName)}
{ }

void TableScan::accept(RelVisitor* visitor) {
    visitor->visit(this);
}

std::vector<std::string>& TableScan::getQualifiedName() {
    return qualifiedName;
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
