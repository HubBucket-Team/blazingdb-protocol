#include <gtest/gtest.h>

#include "api.h"

TEST(ServerTest, API) {
  // TODO(gcca): mock connection to test server
  blazingdb::protocol::UnixSocketConnection connection(
      {"/tmp/socket", std::allocator<char>()});

  blazingdb::protocol::Server server(connection);

  server.handle([](const blazingdb::protocol::Buffer &requestBuffer)
                    -> blazingdb::protocol::Buffer {
    std::cout << requestBuffer.data() << std::endl;

    EXPECT_STREQ("BlazingDB Request",
                 reinterpret_cast<const char *>(requestBuffer.data()));

    return blazingdb::protocol::Buffer(
        reinterpret_cast<const std::uint8_t *>("BlazingDB Response"), 18);
  });

  EXPECT_EQ(true, false);
}
