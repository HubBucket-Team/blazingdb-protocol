#pragma once

#include <vector>
#include <memory>

namespace blazingdb {
namespace protocol {
namespace dto {

class RelNode;

using RelNodePtr = std::shared_ptr<RelNode>;

using VectorRelNodePtr = std::vector<RelNodePtr>;

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
