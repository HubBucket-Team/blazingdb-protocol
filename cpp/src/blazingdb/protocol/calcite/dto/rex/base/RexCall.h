#pragma once

#include "rex/base/RexNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexCall : public RexNode, public virtual RexVisitable {
public:
    RexCall();

    RexCall(KindName kind, TypeName type);

public:
    VectorRexNodePtr& getOperands() override;

public:
    void accept(RexVisitor* visitor) override;

private:
    VectorRexNodePtr operands;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
