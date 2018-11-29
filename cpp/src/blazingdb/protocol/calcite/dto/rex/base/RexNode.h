#pragma once

#include <memory>
#include "rex/base/KindName.h"
#include "rex/base/TypeName.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexNode {
public:
    RexNode();

    RexNode(KindName kind, TypeName type);

public:
    KindName getKindName();

    void setKindName(KindName value);

    TypeName getTypeName();

    void setTypeName(TypeName value);

private:
    KindName kindName;
    TypeName typeName;
};

using RexNodePtr = std::shared_ptr<RexNode>;

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
