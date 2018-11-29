#pragma once

#include "rel/base/RelBase.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelFactory {
public:
    static RelNodePtr createLogicalAggregate();

    static RelNodePtr createLogicalFilter();

    static RelNodePtr createLogicalProject();

    static RelNodePtr createLogicalUnion();

    static RelNodePtr createTableScan();
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
