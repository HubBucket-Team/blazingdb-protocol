#pragma once

#include "rel/base/RelNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelVisitor;

class LogicalAggregate : public RelNode, public virtual RelVisitable {
public:
    LogicalAggregate(std::vector<std::uint64_t>& groups);

    LogicalAggregate(std::vector<std::uint64_t>&& groups);

public:
    void accept(RelVisitor* visitor) override;

public:
    std::vector<std::uint64_t>& getGroups();

private:
    std::vector<std::uint64_t> groups;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
