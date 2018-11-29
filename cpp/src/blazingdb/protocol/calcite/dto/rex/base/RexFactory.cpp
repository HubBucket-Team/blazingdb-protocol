#include "rex/base/RexFactory.h"
#include "rex/base/RexCall.h"

#include <iostream>
namespace blazingdb {
namespace protocol {
namespace dto {

RexNodePtr RexFactory::createRexNode() {
    return std::make_shared<RexNode>();
}

RexNodePtr RexFactory::createRexNode(KindName kind, TypeName type) {
    return std::make_shared<RexNode>(kind, type);
}

RexNodePtr RexFactory::createRexCall() {
    return (RexNodePtr) std::make_shared<RexCall>();
}

RexNodePtr RexFactory::createRexCall(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<RexCall>(kind, type);
}

void RexFactory::addNode(RexNodePtr& collection, RexNodePtr& unit) {
    auto node = std::static_pointer_cast<RexCall>(collection);
    node->getOperands().push_back(unit);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
