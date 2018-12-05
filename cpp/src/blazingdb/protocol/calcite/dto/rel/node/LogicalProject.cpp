#include "rel/node/LogicalProject.h"
#include "rel/visitor/RelVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

LogicalProject::LogicalProject(std::vector<std::string>& columnNames,
                               std::vector<std::uint64_t>& columnIndices)
 : columnNames{columnNames}, columnIndices{columnIndices}
{ }

LogicalProject::LogicalProject(std::vector<std::string>&& columnNames,
                               std::vector<std::uint64_t>&& columnIndices)
 : columnNames{std::move(columnNames)}, columnIndices{std::move(columnIndices)}
{ }

void LogicalProject::accept(RelVisitor* visitor) {
    visitor->visit(this);
}

std::vector<std::string>& LogicalProject::getColumnNames() {
    return columnNames;
}

std::vector<std::uint64_t>& LogicalProject::getColumnIndices() {
    return columnIndices;
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
