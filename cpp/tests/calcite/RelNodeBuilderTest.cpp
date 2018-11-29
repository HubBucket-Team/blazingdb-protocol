#include <gtest/gtest.h>

#include <blazingdb/protocol/calcite/messages/RelNodeBuilder.hpp>

TEST(RelNodeBuilderTest, Main) {
    using namespace blazingdb::protocol::calcite::messages;

    const std::size_t DATA_SIZE = 512;
    std::int8_t data[DATA_SIZE];

    RelNodeBuilder relNodeBilder(data);
    relNodeBilder.Build();
}
