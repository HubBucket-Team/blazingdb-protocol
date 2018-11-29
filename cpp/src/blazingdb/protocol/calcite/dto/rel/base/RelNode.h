#pragma once

#include "rel/base/RelBase.h"
#include "rel/visitor/RelVisitable.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelNode : public virtual RelVisitable {
public:
    RelNodePtr getOperand();

    RelNode* setOperand(RelNodePtr& node);

    RelNode* setOperand(RelNodePtr&& node);

public:
    RelNodePtr getInput(std::size_t i);

    VectorRelNodePtr& getInputs();

public:
    RelNode* addInput(RelNodePtr& node);

    RelNode* addInput(RelNodePtr&& node);

    RelNode* setInput(std::size_t i, RelNodePtr& node);

    RelNode* setInput(std::size_t i, RelNodePtr&& node);

private:
    RelNodePtr operand;
    VectorRelNodePtr inputs;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
