#pragma once

#include "rel/base/RelBase.h"
#include "rex/base/RexBase.h"
#include "rel/visitor/RelVisitable.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelNode : public virtual RelVisitable {
public:
    virtual ~RelNode();

public:
    RexNodePtr& getOperand();

    RelNode* setOperand(RexNodePtr& node);

    RelNode* setOperand(RexNodePtr&& node);

public:
    RelNodePtr getInput(std::size_t i);

    VectorRelNodePtr& getInputs();

public:
    RelNode* addInput(RelNodePtr& node);

    RelNode* addInput(RelNodePtr&& node);

    RelNode* setInput(std::size_t i, RelNodePtr& node);

    RelNode* setInput(std::size_t i, RelNodePtr&& node);

private:
    RexNodePtr operand;
    VectorRelNodePtr inputs;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
