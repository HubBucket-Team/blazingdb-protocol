#include "rel/base/RelNode.h"
#include "rex/base/RexNode.h"
#include "rel/visitor/RelVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

RelNode::~RelNode()
{ }

void RelNode::accept(RelVisitor* visitor) {
    visitor->visit(this);
}

RexNodePtr& RelNode::getOperand() {
    return operand;
}

RelNode* RelNode::setOperand(RexNodePtr& node) {
    operand = node;
    return this;
}

RelNode* RelNode::setOperand(RexNodePtr&& node) {
    operand = std::move(node);
    return this;
}

RelNodePtr RelNode::getInput(std::size_t i) {
    return inputs[i];
}

VectorRelNodePtr& RelNode::getInputs() {
    return inputs;
}

RelNode* RelNode::addInput(RelNodePtr& node) {
    inputs.emplace_back(node);
    return this;
}

RelNode* RelNode::addInput(RelNodePtr&& node) {
    inputs.emplace_back(std::move(node));
    return this;
}

RelNode* RelNode::setInput(std::size_t i, RelNodePtr& node) {
    inputs[i] = node;
    return this;
}

RelNode* RelNode::setInput(std::size_t i, RelNodePtr&& node) {
    inputs[i] = std::move(node);
    return this;
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
