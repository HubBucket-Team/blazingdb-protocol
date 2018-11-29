#include "rel/node/Nodes.h"

namespace blazingdb {
namespace protocol {
namespace dto {

RelNodePtr RelFactory::createLogicalFilter() {
    return (RelNodePtr) std::make_shared<LogicalFilter>();
}

RelNodePtr RelFactory::createLogicalProject() {
    return (RelNodePtr) std::make_shared<LogicalProject>();
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
