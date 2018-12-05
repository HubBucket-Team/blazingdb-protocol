#include "rel/factory/RelFactory.h"
#include "rel/node/Nodes.h"

namespace blazingdb {
namespace protocol {
namespace dto {

RelNodePtr RelFactory::createRelNode() {
    return std::make_shared<RelNode>();
}

RelNodePtr RelFactory::createLogicalAggregate(std::vector<std::uint64_t>& groups) {
    return (RelNodePtr) std::make_shared<LogicalAggregate>(groups);
}

RelNodePtr RelFactory::createLogicalAggregate(std::vector<std::uint64_t>&& groups) {
    return (RelNodePtr) std::make_shared<LogicalAggregate>(std::move(groups));
}

RelNodePtr RelFactory::createLogicalFilter() {
    return (RelNodePtr) std::make_shared<LogicalFilter>();
}

RelNodePtr RelFactory::createLogicalProject(std::vector<std::string>& columnNames,
                                            std::vector<std::uint64_t>& columnIndices) {
    return (RelNodePtr) std::make_shared<LogicalProject>(columnNames, columnIndices);
}

RelNodePtr RelFactory::createLogicalProject(std::vector<std::string>&& columnNames,
                                            std::vector<std::uint64_t>&& columnIndices) {
    return (RelNodePtr) std::make_shared<LogicalProject>(std::move(columnNames), std::move(columnIndices));
}

RelNodePtr RelFactory::createLogicalUnion(bool all) {
    return (RelNodePtr) std::make_shared<LogicalUnion>(all);
}

RelNodePtr RelFactory::createTableScan(std::vector<std::string>& qualifiedName) {
    return (RelNodePtr) std::make_shared<TableScan>(qualifiedName);
}

RelNodePtr RelFactory::createTableScan(std::vector<std::string>&& qualifiedName) {
    return (RelNodePtr) std::make_shared<TableScan>(std::move(qualifiedName));
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
