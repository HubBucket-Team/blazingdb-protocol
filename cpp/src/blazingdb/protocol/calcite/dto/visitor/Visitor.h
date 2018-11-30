#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class RelNode;
class RexNode;
class RelVisitor;
class RexVisitor;

class Visitor {
public:
    static void apply(RelNode* node, RelVisitor* relVisitor, RexVisitor* rexVisitor);

private:
    static void traverse(RelNode* node, RelVisitor* relVisitor, RexVisitor* rexVisitor);

    static void traverse(RexNode* node, RexVisitor* rexVisitor);
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
