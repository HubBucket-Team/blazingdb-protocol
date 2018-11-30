#pragma once

#include "rex/base/RexCall.h"
#include "rex/visitor/RexVisitable.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexVisitor;

class FieldAccess : public RexCall, public virtual RexVisitable {
public:
    FieldAccess(KindName kind, TypeName type);

public:
    void accept(RexVisitor* visitor) override;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
