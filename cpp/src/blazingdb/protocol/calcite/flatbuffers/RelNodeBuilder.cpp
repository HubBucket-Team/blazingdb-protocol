#include "RelNodeBuilder.hpp"

#include "../../message/generated/all_generated.h"

namespace blazingdb {
namespace protocol {
namespace calcite {
namespace flatbuffers {

class RelNodeBuilder::RelNodeBuilderImpl {
public:
    RelNodeBuilderImpl(const Buffer<std::int8_t> &buffer) : buffer_{buffer} {}
    inline virtual ~RelNodeBuilderImpl() = default;

    void Build() const {}

private:
    const Buffer<std::int8_t> buffer_;
};

RelNodeBuilder::RelNodeBuilder(const Buffer<std::int8_t> &buffer)
    : impl_{new RelNodeBuilderImpl{buffer}} {}

RelNodeBuilder::~RelNodeBuilder() = default;

void RelNodeBuilder::Build() const { impl_->Build(); }

}  // namespace flatbuffers
}  // namespace calcite
}  // namespace protocol
}  // namespace blazingdb
