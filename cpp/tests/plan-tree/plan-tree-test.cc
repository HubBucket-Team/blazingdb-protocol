#include <gtest/gtest.h>

#include "tree_generated.h"

TEST(PlanTreeTest, Hi) {
  using namespace plan::tree;

  // serializing union node
  flatbuffers::FlatBufferBuilder unionBuilder(0);
  auto unionNode = CreateUnionNodeDirect(unionBuilder, true);
  unionBuilder.Finish(unionNode);
  auto unionBuffer = unionBuilder.GetBufferPointer();
  auto unionBufferSize = unionBuilder.GetSize();

  flatbuffers::FlatBufferBuilder builder(0);

  // put union buffer on child
  auto unionOffset = builder.CreateVector(unionBuffer, unionBufferSize);
  auto unionChild = CreateChild(builder,
      NodeType::NodeType_Union, unionOffset);

  // put child on root
  std::vector<flatbuffers::Offset<Child>> rootChildrenVector;
  rootChildrenVector.push_back(unionChild);
  auto rootChildren = builder.CreateVector(rootChildrenVector);

  auto root = CreateRootNode(builder, rootChildren);
  builder.Finish(root);

  // create root buffer
  auto rootBuffer = builder.GetBufferPointer();
  auto rootBufferSize = builder.GetSize();

  // ----------------------------------------------------

  {
    auto root = GetRootNode(rootBuffer);
    auto result = EnumNameNodeType(root->children()->Get(0)->type());
    EXPECT_STREQ("Union", result);

    const UnionNode *unionNode = flatbuffers::GetRoot<UnionNode>(
        root->children()->Get(0)->data()->Data());
    EXPECT_EQ(true, unionNode->all());
  }
}
