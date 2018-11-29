#pragma once

#include "rex/base/RexNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexCall : public RexNode {
public:
    RexCall();

    RexCall(KindName kind, TypeName type);

public:
    VectorRexNodePtr& getOperands() override;

private:
    VectorRexNodePtr operands;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
