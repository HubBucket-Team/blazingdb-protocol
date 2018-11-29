#pragma once

namespace blazingdb {
namespace protocol {
namespace dto {

class RelVisitor;

class RelVisitable {
public:
    virtual ~RelVisitable()
    { }

public:
    virtual void accept(RelVisitor* visitor) = 0;
};

}  // namespace dto
}  // namespace protocol
}  // namespace blazingdb
