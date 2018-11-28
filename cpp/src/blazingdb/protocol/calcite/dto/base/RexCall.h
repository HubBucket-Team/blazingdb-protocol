#pragma once

#include <vector>
#include "base/RexNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

using VectorRexNodePtr = std::vector<RexNodePtr>;

class RexCall : public RexNode {
public:
    RexCall();

    RexCall(KindName kind, TypeName type);

public:
    VectorRexNodePtr& getOperands();

private:
    VectorRexNodePtr operands;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
