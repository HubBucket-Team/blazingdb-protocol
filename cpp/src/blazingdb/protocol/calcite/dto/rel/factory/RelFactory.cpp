#include "rel/factory/RelFactory.h"
#include "rel/node/Nodes.h"

namespace blazingdb {
namespace protocol {
namespace dto {

RelNodePtr RelFactory::createRelNode() {
    return std::make_shared<RelNode>();
}

RelNodePtr RelFactory::createLogicalAggregate() {
    return (RelNodePtr) std::make_shared<LogicalAggregate>();
}

RelNodePtr RelFactory::createLogicalFilter() {
    return (RelNodePtr) std::make_shared<LogicalFilter>();
}

RelNodePtr RelFactory::createLogicalProject() {
    return (RelNodePtr) std::make_shared<LogicalProject>();
}

RelNodePtr RelFactory::createLogicalUnion() {
    return (RelNodePtr) std::make_shared<LogicalUnion>();
}

RelNodePtr RelFactory::createTableScan() {
    return (RelNodePtr) std::make_shared<TableScan>();
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
