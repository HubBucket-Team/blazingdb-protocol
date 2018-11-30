#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class Literal;
class TableInputRef;

class RexVisitor {
public:
    virtual ~RexVisitor()
    { }

public:
    virtual void visit(Literal* node) = 0;

    virtual void visit(TableInputRef* node) = 0;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
