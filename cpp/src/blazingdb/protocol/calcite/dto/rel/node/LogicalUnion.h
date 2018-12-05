#pragma once

#include "rel/base/RelNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelVisitor;

class LogicalUnion : public RelNode, public virtual RelVisitable {
public:
    LogicalUnion(bool all);

public:
    void accept(RelVisitor* visitor) override;

public:
    bool getAll() const;

private:
    const bool all;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
