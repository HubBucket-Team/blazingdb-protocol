#pragma once

#include "rel/base/RelNode.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelVisitor;

class LogicalProject : public RelNode, public virtual RelVisitable {
public:
    LogicalProject(std::vector<std::string>& columnNames,
                   std::vector<std::uint64_t>& columnIndices);

    LogicalProject(std::vector<std::string>&& columnNames,
                   std::vector<std::uint64_t>&& columnIndices);

public:
    void accept(RelVisitor* visitor) override;

public:
    std::vector<std::string>& getColumnNames();

    std::vector<std::uint64_t>& getColumnIndices();

private:
    std::vector<std::string> columnNames;
    std::vector<std::uint64_t> columnIndices;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
