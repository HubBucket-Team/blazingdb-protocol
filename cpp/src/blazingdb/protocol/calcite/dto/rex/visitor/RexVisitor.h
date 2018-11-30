#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class RexCall;
class CorrelVariable;
class DynamicParam;
class FieldAccess;
class InputRef;
class Literal;
class LocalRef;
class Over;
class PatternFieldRef;
class RangeRef;
class SubQuery;
class TableInputRef;
class Variable;

class RexVisitor {
public:
    virtual ~RexVisitor()
    { }

public:
    virtual void visit(RexCall* node) = 0;

    virtual void visit(CorrelVariable* node) = 0;

    virtual void visit(DynamicParam* node) = 0;

    virtual void visit(FieldAccess* node) = 0;

    virtual void visit(InputRef* node) = 0;

    virtual void visit(Literal* node) = 0;

    virtual void visit(LocalRef* node) = 0;

    virtual void visit(Over* node) = 0;

    virtual void visit(PatternFieldRef* node) = 0;

    virtual void visit(RangeRef* node) = 0;

    virtual void visit(SubQuery* node) = 0;

    virtual void visit(TableInputRef* node) = 0;

    virtual void visit(Variable* node) = 0;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
