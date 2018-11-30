#include "rex/factory/RexFactory.h"
#include "rex/node/Nodes.h"

namespace blazingdb {
namespace protocol {
namespace dto {

void RexFactory::addNode(RexNodePtr& collection, RexNodePtr& unit) {
    collection->getOperands().push_back(unit);
}

RexNodePtr RexFactory::createRexCall(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<RexCall>(kind, type);
}

RexNodePtr RexFactory::createCorrelVariable(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<CorrelVariable>(kind, type);
}

RexNodePtr RexFactory::createDynamicParam(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<DynamicParam>(kind, type);
}

RexNodePtr RexFactory::createFieldAccess(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<FieldAccess>(kind, type);
}

RexNodePtr RexFactory::createInputRef(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<InputRef>(kind, type);
}

RexNodePtr RexFactory::createLiteral(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<Literal>(kind, type);
}

RexNodePtr RexFactory::createLocalRef(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<LocalRef>(kind, type);
}

RexNodePtr RexFactory::createOver(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<Over>(kind, type);
}

RexNodePtr RexFactory::createPatternFieldRef(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<PatternFieldRef>(kind, type);
}

RexNodePtr RexFactory::createRangeRef(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<RangeRef>(kind, type);
}

RexNodePtr RexFactory::createSubQuery(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<SubQuery>(kind, type);
}

RexNodePtr RexFactory::createTableInputRef(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<TableInputRef>(kind, type);
}

RexNodePtr RexFactory::createVariable(KindName kind, TypeName type) {
    return (RexNodePtr) std::make_shared<Variable>(kind, type);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
