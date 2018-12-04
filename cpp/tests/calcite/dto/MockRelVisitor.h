#pragma once

#include "gmock/gmock.h"
#include "rel/visitor/RelVisitor.h"

namespace LogicalPlan = blazingdb::protocol::dto;

struct MockRelVisitor : public LogicalPlan::RelVisitor {
    MOCK_METHOD1(visit, void(LogicalPlan::RelNode*));

    MOCK_METHOD1(visit, void(LogicalPlan::LogicalAggregate*));

    MOCK_METHOD1(visit, void(LogicalPlan::LogicalFilter*));

    MOCK_METHOD1(visit, void(LogicalPlan::LogicalProject*));

    MOCK_METHOD1(visit, void(LogicalPlan::LogicalUnion*));

    MOCK_METHOD1(visit, void(LogicalPlan::TableScan*));
};
