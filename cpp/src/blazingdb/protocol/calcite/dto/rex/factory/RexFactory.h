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

    static RexNodePtr createLiteral(KindName kind, TypeName type);

    static RexNodePtr createTableInputRef(KindName kind, TypeName type);
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
