#include "rel/node/LogicalAggregate.h"
#include "rel/visitor/RelVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

LogicalAggregate::LogicalAggregate(std::vector<std::uint64_t>& groups)
 : groups{groups}
{ }

LogicalAggregate::LogicalAggregate(std::vector<std::uint64_t>&& groups)
 : groups{std::move(groups)}
{ }

void LogicalAggregate::accept(RelVisitor* visitor) {
    visitor->visit(this);
}

std::vector<std::uint64_t>& LogicalAggregate::getGroups() {
    return groups;
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
