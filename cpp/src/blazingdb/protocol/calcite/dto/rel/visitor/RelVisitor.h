#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class LogicalProject;
class LogicalFilter;
/*
class TableScan;
class LogicalAggregate;
class LogicalUnion;
*/

class RelVisitor {
public:
    virtual void visit(LogicalFilter* node) = 0;

    virtual void visit(LogicalProject* node) = 0;

/*
    virtual void visit(TableScan* node) = 0;

    virtual void visit(LogicalAggregate* node) = 0;

    virtual void visit(LogicalUnion* node) = 0;
    */
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
