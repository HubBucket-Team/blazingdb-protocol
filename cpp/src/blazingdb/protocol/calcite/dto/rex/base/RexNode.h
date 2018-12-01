#pragma once

#include "rex/base/RexBase.h"
#include "rex/base/RexData.h"
#include "rex/base/KindName.h"
#include "rex/base/TypeName.h"
#include "rex/visitor/RexVisitable.h"

namespace blazingdb {
namespace protocol {
namespace dto {

class RexNode : public virtual RexVisitable {
public:
    RexNode();

    virtual ~RexNode();

public:
    RexNode(KindName kind, TypeName type);

public:
    KindName getKindName();

    void setKindName(KindName value);

    TypeName getTypeName();

    void setTypeName(TypeName value);

public:
    RexData& getRexData();

    void setRexData(RexData& data);

    void setRexData(RexData&& data);

public:
    virtual VectorRexNodePtr& getOperands() = 0;

private:
    KindName kindName;
    TypeName typeName;

private:
    RexData rexData;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
