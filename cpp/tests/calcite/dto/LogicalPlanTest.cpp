#include "gtest/gtest.h"
#include "LogicalPlan.h"
#include "MockRelVisitor.h"
#include "MockRexVisitor.h"

using ::testing::An;
using ::testing::Sequence;

namespace LogicalPlan = blazingdb::protocol::dto;

struct LogicalPlanTest : public ::testing::Test {
    LogicalPlanTest() {
    }

    virtual ~LogicalPlanTest() {
    }

    virtual void SetUp() {
    }

    virtual void TearDown() {
    }

    Sequence sequence;
    MockRelVisitor mockRelVisitor;
    MockRexVisitor mockRexVisitor;
};


TEST_F(LogicalPlanTest, RelVisitorTest) {
    auto relNode0 = LogicalPlan::RelFactory::createLogicalFilter();
    auto relNode1 = LogicalPlan::RelFactory::createLogicalProject();
    auto relNode2 = LogicalPlan::RelFactory::createLogicalAggregate();
    auto relNode3 = LogicalPlan::RelFactory::createLogicalUnion();
    auto relNode4 = LogicalPlan::RelFactory::createTableScan();
    auto relNode5 = LogicalPlan::RelFactory::createLogicalFilter();
    auto relNode6 = LogicalPlan::RelFactory::createTableScan();
    auto relNode7 = LogicalPlan::RelFactory::createLogicalAggregate();
    auto relNode8 = LogicalPlan::RelFactory::createLogicalProject();
    auto relNode9 = LogicalPlan::RelFactory::createLogicalFilter();

    relNode0->addInput(relNode1);
    relNode0->addInput(relNode2);
    relNode0->addInput(relNode3);

    relNode4->addInput(relNode5);
    relNode4->addInput(relNode6);

    relNode7->addInput(relNode8);

    relNode9->addInput(relNode7);
    relNode9->addInput(relNode4);
    relNode9->addInput(relNode0);

    // relNode8
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalProject*>()))
            .InSequence(sequence);
    // relNode7
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalAggregate*>()))
            .InSequence(sequence);
    // relNode5
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalFilter*>()))
            .InSequence(sequence);
    // relNode6
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::TableScan*>()))
            .InSequence(sequence);
    // relNode4
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::TableScan*>()))
            .InSequence(sequence);
    // relNode1
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalProject*>()))
            .InSequence(sequence);
    // relNode2
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalAggregate*>()))
            .InSequence(sequence);
    // relNode3
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalUnion*>()))
            .InSequence(sequence);
    // relNode0
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalFilter*>()))
            .InSequence(sequence);
    // relNode9
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalFilter*>()))
            .InSequence(sequence);

    LogicalPlan::Visitor::apply(relNode9.get(), &mockRelVisitor, nullptr);
}


TEST_F(LogicalPlanTest, RexVisitorTest) {
    using LogicalPlan::KindName;
    using LogicalPlan::TypeName;

    auto relNode = LogicalPlan::RelFactory::createLogicalFilter();
    auto rexNode0 = LogicalPlan::RexFactory::createVariable(KindName::AND, TypeName::DECIMAL);
    auto rexNode1 = LogicalPlan::RexFactory::createTableInputRef(KindName::AND, TypeName::DECIMAL);
    auto rexNode2 = LogicalPlan::RexFactory::createRexCall(KindName::AND, TypeName::DECIMAL);
    auto rexNode3 = LogicalPlan::RexFactory::createCorrelVariable(KindName::AND, TypeName::DECIMAL);
    auto rexNode4 = LogicalPlan::RexFactory::createDynamicParam(KindName::AND, TypeName::DECIMAL);
    auto rexNode5 = LogicalPlan::RexFactory::createFieldAccess(KindName::AND, TypeName::DECIMAL);
    auto rexNode6 = LogicalPlan::RexFactory::createInputRef(KindName::AND, TypeName::DECIMAL);
    auto rexNode7 = LogicalPlan::RexFactory::createLocalRef(KindName::AND, TypeName::DECIMAL);
    auto rexNode8 = LogicalPlan::RexFactory::createOver(KindName::AND, TypeName::DECIMAL);

    LogicalPlan::RexFactory::addNode(rexNode3, rexNode2);
    LogicalPlan::RexFactory::addNode(rexNode3, rexNode4);

    LogicalPlan::RexFactory::addNode(rexNode1, rexNode0);
    LogicalPlan::RexFactory::addNode(rexNode1, rexNode3);

    LogicalPlan::RexFactory::addNode(rexNode7, rexNode8);
    LogicalPlan::RexFactory::addNode(rexNode6, rexNode7);

    LogicalPlan::RexFactory::addNode(rexNode5, rexNode1);
    LogicalPlan::RexFactory::addNode(rexNode5, rexNode6);

    relNode->setOperand(rexNode5);

    // rexNode0
    EXPECT_CALL(mockRelVisitor, visit(An<LogicalPlan::LogicalFilter*>()))
            .InSequence(sequence);
    // rexNode0
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::Variable*>()))
            .InSequence(sequence);
    // rexNode2
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::RexCall*>()))
            .InSequence(sequence);
    // rexNode4
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::DynamicParam*>()))
            .InSequence(sequence);
    // rexNode3
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::CorrelVariable*>()))
            .InSequence(sequence);
    // rexNode1
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::TableInputRef*>()))
            .InSequence(sequence);
    // rexNode8
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::Over*>()))
            .InSequence(sequence);
    // rexNode7
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::LocalRef*>()))
            .InSequence(sequence);
    // rexNode6
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::InputRef*>()))
            .InSequence(sequence);
    // rexNode5
    EXPECT_CALL(mockRexVisitor, visit(An<LogicalPlan::FieldAccess*>()))
            .InSequence(sequence);

    LogicalPlan::Visitor::apply(relNode.get(), &mockRelVisitor, &mockRexVisitor);
}
