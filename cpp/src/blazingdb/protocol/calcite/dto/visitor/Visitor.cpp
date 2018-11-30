#include "visitor/Visitor.h"
#include "rel/base/RelNode.h"
#include "rex/base/RexNode.h"
#include "rel/visitor/RelVisitor.h"
#include "rex/visitor/RexVisitor.h"

namespace blazingdb {
namespace protocol {
namespace dto {

void Visitor::apply(RelNode* node, RelVisitor* relVisitor, RexVisitor* rexVisitor) {
    traverse(node, relVisitor, rexVisitor);
}

void Visitor::traverse(RelNode* node, RelVisitor* relVisitor, RexVisitor* rexVisitor) {
    auto& inputs = node->getInputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        traverse(inputs[i].get(), relVisitor, rexVisitor);
    }

    node->accept(relVisitor);
    traverse(node->getOperand().get(), rexVisitor);
}

void Visitor::traverse(RexNode* node, RexVisitor* rexVisitor) {
    if (node == nullptr) {
        return;
    }

    auto& operands = node->getOperands();
    for (std::size_t i = 0; i < operands.size(); ++i) {
        traverse(operands[i].get(), rexVisitor);
    }

    node->accept(rexVisitor);
}

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
