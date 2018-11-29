#include "rex/factory/RexFactory.h"
#include "rex/node/Nodes.h"

namespace blazingdb {
namespace protocol {
namespace dto {

void RexFactory::addNode(RexNodePtr& collection, RexNodePtr& unit) {
    collection->getOperands().push_back(unit);
}

RexNodePtr RexFactory::createLiteral(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<Literal>(kind, type);
}

RexNodePtr RexFactory::createTableInputRef(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<TableInputRef>(kind, type);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
