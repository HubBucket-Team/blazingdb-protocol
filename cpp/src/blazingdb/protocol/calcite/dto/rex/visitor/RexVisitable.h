#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class RexVisitor;

class RexVisitable {
public:
    virtual ~RexVisitable()
    { }

public:
    virtual void accept(RexVisitor* visitor) = 0;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
