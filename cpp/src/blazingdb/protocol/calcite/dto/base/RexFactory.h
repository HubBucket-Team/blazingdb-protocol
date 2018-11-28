#pragma once

#include "base/RexNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexFactory {
public:
    static RexNodePtr createRexNode();

    static RexNodePtr createRexNode(KindName kind, TypeName type);

    static RexNodePtr createRexCall();

    static RexNodePtr createRexCall(KindName kind, TypeName type);

    static void addNode(RexNodePtr& collection, RexNodePtr& unit);
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
