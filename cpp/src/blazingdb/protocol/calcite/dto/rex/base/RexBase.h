#pragma once

#include <memory>
#include <vector>

namespace blazingdb {
namespace protocol {
namespace dto {

class RexNode;

using RexNodePtr = std::shared_ptr<RexNode>;

using VectorRexNodePtr = std::vector<RexNodePtr>;

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
