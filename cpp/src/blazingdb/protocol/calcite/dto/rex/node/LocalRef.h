#pragma once

#include "rex/base/RexCall.h"
#include "rex/visitor/RexVisitable.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexVisitor;

class LocalRef : public RexCall, public virtual RexVisitable {
public:
    LocalRef(KindName kind, TypeName type);

public:
    void accept(RexVisitor* visitor) override;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
