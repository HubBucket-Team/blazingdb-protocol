#pragma once

#include "rel/base/RelNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelVisitor;

class TableScan : public RelNode, public virtual RelVisitable {
public:
    TableScan(std::vector<std::string>& qualifiedName);

    TableScan(std::vector<std::string>&& qualifiedName);

public:
    void accept(RelVisitor* visitor) override;

public:
    std::vector<std::string>& getQualifiedName();

private:
    std::vector<std::string> qualifiedName;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
