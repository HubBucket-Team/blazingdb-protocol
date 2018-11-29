#ifndef BLAZINGDB_PROTOCOL_CALCITE_FLATBUFFERS_RELNODEBUILDER_H_
#define BLAZINGDB_PROTOCOL_CALCITE_FLATBUFFERS_RELNODEBUILDER_H_

#include <cstdint>
#include <memory>
#include <type_traits>

namespace blazingdb {
namespace protocol {
namespace calcite {
namespace flatbuffers {

template <class T, std::ptrdiff_t diff = -1>
class Buffer {
public:
    using pointer         = T *const;
    using const_pointer   = const T *const;
    using reference       = T &;
    using const_reference = const T &;
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::size_t;

    constexpr Buffer(const_pointer data, const size_type size) noexcept
        : data_{data}, size_{size} {}

    template <std::size_t size>
    constexpr Buffer(const value_type (&data)[size]) noexcept
        : Buffer(data, size) {}

    constexpr const_pointer data() const noexcept { return data_; }
    constexpr size_type     size() const noexcept { return size_; }

private:
    const_pointer     data_;
    const std::size_t size_;
};

class NodeBuilder {
public:
    inline NodeBuilder()          = default;
    inline virtual ~NodeBuilder() = default;
    virtual void Build() const    = 0;

private:
    NodeBuilder(const NodeBuilder &) = delete;
    NodeBuilder(NodeBuilder &&)      = delete;
    void operator=(const NodeBuilder &) = delete;
    void operator=(NodeBuilder &&) = delete;
};

class RelNodeBuilder : public NodeBuilder {
public:
    virtual ~RelNodeBuilder();

    RelNodeBuilder(const Buffer<std::int8_t> &);
    void Build() const final;

private:
    class RelNodeBuilderImpl;
    const std::unique_ptr<RelNodeBuilderImpl> impl_;
};

}  // namespace flatbuffers
}  // namespace calcite
}  // namespace protocol
}  // namespace blazingdb

#endif
