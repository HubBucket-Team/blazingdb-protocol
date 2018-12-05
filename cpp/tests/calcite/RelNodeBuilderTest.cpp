#include <blazingdb/protocol/calcite/messages/RelNodeBuilder.hpp>
#include <typeinfo>
#include <gtest/gtest.h>

#include "utils.hpp"
#include "LogicalPlan.h"

TEST(RelNodeBuilderTest, SingleCreation) {
    using namespace com::blazingdb::protocol::calcite::plan::messages;

    auto leftTableScanNodeDetachedBuffer =
        factory::CreateTableScanNodeDetachedBuffer({"left", "table"});

    auto leftTableScanNode =
        flatbuffers::GetRoot<RelNode>(leftTableScanNodeDetachedBuffer.data());

    EXPECT_EQ(RelNodeType_TableScan, leftTableScanNode->type());
    EXPECT_NE(nullptr, leftTableScanNode->data());

    auto leftTableScan =
        flatbuffers::GetRoot<TableScan>(leftTableScanNode->data()->Data());

    EXPECT_EQ("left", leftTableScan->qualifiedName()->GetAsString(0)->str());
    EXPECT_EQ("table", leftTableScan->qualifiedName()->GetAsString(1)->str());
}

TEST(RelNodeBuilderTest, OnceNestedCreation) {
    using namespace com::blazingdb::protocol::calcite::plan::messages;

    auto leftTableScanNodeDetachedBuffer =
        factory::CreateTableScanNodeDetachedBuffer({"left", "table"});
    auto rightTableScanNodeDetachedBuffer =
        factory::CreateTableScanNodeDetachedBuffer({"right", "table"});

    auto logicalUnionNodeDetachedBuffer =
        factory::CreateLogicalUnionNodeDetachedBuffer(
            true,
            leftTableScanNodeDetachedBuffer,
            rightTableScanNodeDetachedBuffer);

    auto logicalUnionNode =
        flatbuffers::GetRoot<RelNode>(logicalUnionNodeDetachedBuffer.data());
    EXPECT_EQ(RelNodeType_LogicalUnion, logicalUnionNode->type());
    EXPECT_NE(nullptr, logicalUnionNode->data());

    auto logicalUnion =
        flatbuffers::GetRoot<LogicalUnion>(logicalUnionNode->data()->Data());
    EXPECT_TRUE(logicalUnion->all());

    EXPECT_NE(nullptr, logicalUnionNode->inputs());
    EXPECT_EQ(2, logicalUnionNode->inputs()->Length());

    auto leftTableScanNode = logicalUnionNode->inputs()->Get(0);
    EXPECT_EQ(RelNodeType_TableScan, leftTableScanNode->type());
    EXPECT_EQ(nullptr, leftTableScanNode->inputs());

    auto leftTableScan =
        flatbuffers::GetRoot<TableScan>(leftTableScanNode->data()->Data());
    EXPECT_EQ(2, leftTableScan->qualifiedName()->Length());
    EXPECT_EQ("left", leftTableScan->qualifiedName()->GetAsString(0)->str());
    EXPECT_EQ("table", leftTableScan->qualifiedName()->GetAsString(1)->str());

    auto rightTableScanNode = logicalUnionNode->inputs()->Get(1);
    EXPECT_EQ(RelNodeType_TableScan, rightTableScanNode->type());
    EXPECT_EQ(nullptr, rightTableScanNode->inputs());

    auto rightTableScan =
        flatbuffers::GetRoot<TableScan>(rightTableScanNode->data()->Data());
    EXPECT_EQ(2, rightTableScan->qualifiedName()->Length());
    EXPECT_EQ("right", rightTableScan->qualifiedName()->GetAsString(0)->str());
    EXPECT_EQ("table", rightTableScan->qualifiedName()->GetAsString(1)->str());
}

TEST(RelNodeBuilderTest, TwiceNestedCreation) {
    using namespace com::blazingdb::protocol::calcite::plan::messages;

    auto leftTableScanNodeDetachedBuffer =
        factory::CreateTableScanNodeDetachedBuffer({"left", "LEFT"});
    auto rightTableScanNodeDetachedBuffer =
        factory::CreateTableScanNodeDetachedBuffer({"right", "RIGHT"});

    auto leftLogicalProjectDetachedBuffers =
        factory::CreateLogicalProjectNodeDetachedBuffer(
            {"COL3"}, {2}, leftTableScanNodeDetachedBuffer);
    auto rightLogicalProjectDetachedBuffers =
        factory::CreateLogicalProjectNodeDetachedBuffer(
            {"COL2"}, {1}, rightTableScanNodeDetachedBuffer);

    auto logicalUnionNodeDetachedBuffer =
        factory::CreateLogicalUnionNodeDetachedBuffer(
            true,
            leftLogicalProjectDetachedBuffers,
            rightLogicalProjectDetachedBuffers);

    auto logicalUnionNode =
        flatbuffers::GetRoot<RelNode>(logicalUnionNodeDetachedBuffer.data());
    EXPECT_EQ(RelNodeType_LogicalUnion, logicalUnionNode->type());
    EXPECT_NE(nullptr, logicalUnionNode->data());

    auto logicalUnion =
        flatbuffers::GetRoot<LogicalUnion>(logicalUnionNode->data()->Data());
    EXPECT_TRUE(logicalUnion->all());
}

TEST(RelNodeBuilderTest, RelNodes) {
    namespace LogicalPlan = blazingdb::protocol::dto;
    namespace Builder = blazingdb::protocol::calcite::messages;
    using namespace com::blazingdb::protocol::calcite::plan::messages;

    std::vector<std::string> columnNames{"Customer", "User", "Store"};
    std::vector<std::uint64_t> columnIndices{1, 2, 3};

    auto TableScanDetachedBuffer =
        factory::CreateTableScanNodeDetachedBuffer(columnNames);
    auto LogicalProjectDetachedBuffer =
        factory::CreateLogicalProjectNodeDetachedBuffer(columnNames,
                                                        columnIndices,
                                                        TableScanDetachedBuffer);

    auto logicalUnionNodeDetachedBuffer =
        factory::CreateLogicalUnionNodeDetachedBuffer(
            true,
            TableScanDetachedBuffer,
            LogicalProjectDetachedBuffer);

    Builder::Buffer<std::uint8_t> buffer(logicalUnionNodeDetachedBuffer.data(),
                                         logicalUnionNodeDetachedBuffer.size());

    Builder::RelNodeBuilder relNodeBilder(buffer);
    auto relNode = relNodeBilder.Build();

    // Verify LogicalUnion
    {
        EXPECT_TRUE(typeid(*relNode) == typeid(LogicalPlan::LogicalUnion));
        auto node = std::dynamic_pointer_cast<LogicalPlan::LogicalUnion>(relNode);

        EXPECT_TRUE(node->getAll());
        EXPECT_TRUE(node->getInputs().size() == 2);
    }

    // Verify TableScan
    {
        ASSERT_TRUE(typeid(*(relNode->getInputs()[0])) == typeid(LogicalPlan::TableScan));
        auto node = std::dynamic_pointer_cast<LogicalPlan::TableScan>(relNode->getInputs()[0]);

        EXPECT_TRUE(node->getInputs().size() == 0);
        EXPECT_TRUE(node->getQualifiedName().size() == columnNames.size());
        for (std::size_t i = 0; i < columnNames.size(); ++i) {
            EXPECT_TRUE(columnNames[i] == node->getQualifiedName()[i]);
        }
    }

    // Verify LogicalProject
    {
        ASSERT_TRUE(typeid(*(relNode->getInputs()[1])) == typeid(LogicalPlan::LogicalProject));
        auto node = std::dynamic_pointer_cast<LogicalPlan::LogicalProject>(relNode->getInputs()[1]);

        EXPECT_TRUE(node->getInputs().size() == 0);
        EXPECT_TRUE(node->getColumnNames().size() == columnNames.size());
        EXPECT_TRUE(node->getColumnIndices().size() == columnIndices.size());

        for (std::size_t i = 0; i < columnNames.size(); ++i) {
            EXPECT_TRUE(columnNames[i] == node->getColumnNames()[i]);
        }
        for (std::size_t i = 0; i < columnIndices.size(); ++i) {
            EXPECT_TRUE(columnIndices[i] == node->getColumnIndices()[i]);
        }
    }
}
