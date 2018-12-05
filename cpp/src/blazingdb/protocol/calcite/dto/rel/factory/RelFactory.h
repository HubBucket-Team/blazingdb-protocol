#pragma once

#include "rel/base/RelBase.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelFactory {
public:
    static RelNodePtr createRelNode();

    static RelNodePtr createLogicalAggregate(std::vector<std::uint64_t>& groups);

    static RelNodePtr createLogicalAggregate(std::vector<std::uint64_t>&& groups = {});

    static RelNodePtr createLogicalFilter();

    static RelNodePtr createLogicalProject(std::vector<std::string>& columnNames,
                                           std::vector<std::uint64_t>& columnIndices);

    static RelNodePtr createLogicalProject(std::vector<std::string>&& columnNames = {},
                                           std::vector<std::uint64_t>&& columnIndices = {});

    static RelNodePtr createLogicalUnion(bool all = false);

    static RelNodePtr createTableScan(std::vector<std::string>& qualifiedName);

    static RelNodePtr createTableScan(std::vector<std::string>&& qualifiedName = {});
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
