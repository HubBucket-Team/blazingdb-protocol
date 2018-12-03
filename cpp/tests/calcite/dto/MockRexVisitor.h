#pragma once

#include "gmock/gmock.h"
#include "rex/visitor/RexVisitor.h"

namespace LogicalPlan = blazingdb::protocol::dto;

struct MockRexVisitor : public LogicalPlan::RexVisitor {
    MOCK_METHOD1(visit, void(LogicalPlan::RexCall*));

    MOCK_METHOD1(visit, void(LogicalPlan::CorrelVariable*));

    MOCK_METHOD1(visit, void(LogicalPlan::DynamicParam*));

    MOCK_METHOD1(visit, void(LogicalPlan::FieldAccess*));

    MOCK_METHOD1(visit, void(LogicalPlan::InputRef*));

    MOCK_METHOD1(visit, void(LogicalPlan::Literal*));

    MOCK_METHOD1(visit, void(LogicalPlan::LocalRef*));

    MOCK_METHOD1(visit, void(LogicalPlan::Over*));

    MOCK_METHOD1(visit, void(LogicalPlan::PatternFieldRef*));

    MOCK_METHOD1(visit, void(LogicalPlan::RangeRef*));

    MOCK_METHOD1(visit, void(LogicalPlan::SubQuery*));

    MOCK_METHOD1(visit, void(LogicalPlan::TableInputRef*));

    MOCK_METHOD1(visit, void(LogicalPlan::Variable*));
};
