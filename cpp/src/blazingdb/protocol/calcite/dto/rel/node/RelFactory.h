#pragma once

#include "rel/base/RelBase.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RelFactory {
public:
    static RelNodePtr createLogicalFilter();

    static RelNodePtr createLogicalProject();
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
