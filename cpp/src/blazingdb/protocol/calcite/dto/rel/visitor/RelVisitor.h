#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class RelNode;
class LogicalAggregate;
class LogicalFilter;
class LogicalProject;
class LogicalUnion;
class TableScan;

class RelVisitor {
public:
    virtual ~RelVisitor()
    { }

public:
    virtual void visit(RelNode* node) = 0;

    virtual void visit(LogicalAggregate* node) = 0;

    virtual void visit(LogicalFilter* node) = 0;

    virtual void visit(LogicalProject* node) = 0;

    virtual void visit(LogicalUnion* node) = 0;

    virtual void visit(TableScan* node) = 0;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
