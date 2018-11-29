#pragma once

#include "rel/base/RelNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelVisitor;

class TableScan : public RelNode, public virtual RelVisitable {
public:
    void accept(RelVisitor* visitor) override;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
