#pragma once

#include "rex/base/RexBase.h"
#include "rex/base/KindName.h"
#include "rex/base/TypeName.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexFactory {
public:
    static void addNode(RexNodePtr& collection, RexNodePtr& unit);

    static RexNodePtr createRexCall(KindName kind, TypeName type);

    static RexNodePtr createCorrelVariable(KindName kind, TypeName type);

    static RexNodePtr createDynamicParam(KindName kind, TypeName type);

    static RexNodePtr createFieldAccess(KindName kind, TypeName type);

    static RexNodePtr createInputRef(KindName kind, TypeName type);

    static RexNodePtr createLiteral(KindName kind, TypeName type);

    static RexNodePtr createLocalRef(KindName kind, TypeName type);

    static RexNodePtr createOver(KindName kind, TypeName type);

    static RexNodePtr createPatternFieldRef(KindName kind, TypeName type);

    static RexNodePtr createRangeRef(KindName kind, TypeName type);

    static RexNodePtr createSubQuery(KindName kind, TypeName type);

    static RexNodePtr createTableInputRef(KindName kind, TypeName type);

    static RexNodePtr createVariable(KindName kind, TypeName type);
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
