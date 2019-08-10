#include "messages.h"

#include <iostream>
#include <sstream>

#define BLAZING_NORETURN __attribute__((__noreturn__))
#define BLAZING_ABORT(_message)                                                \
    do {                                                                       \
        std::stringstream ss{std::ios_base::out | std::ios_base::in};          \
        ss << __FILE__ << ':' << __LINE__ << ": " << (_message) << std::endl;  \
        std::cerr << ss.str();                                                 \
        std::exit(-1);                                                         \
    } while (0)
#define BLAZING_INTERFACE(Kind)                                                \
public:                                                                        \
    virtual ~Kind() = default;                                                 \
                                                                               \
protected:                                                                     \
    explicit Kind() = default;                                                 \
                                                                               \
private:                                                                       \
    Kind(const Kind &)  = delete;                                              \
    Kind(const Kind &&) = delete;                                              \
    void operator=(const Kind &) = delete;                                     \
    void operator=(const Kind &&) = delete
#define BLAZING_CONCRETE(Kind)                                                 \
private:                                                                       \
    Kind(const Kind &)  = delete;                                              \
    Kind(const Kind &&) = delete;                                              \
    void operator=(const Kind &) = delete;                                     \
    void operator=(const Kind &&) = delete

namespace blazingdb {
namespace protocol {
namespace orchestrator {

// DDLCreateTableRequestMessage

//   Node table schema

class NodeTableSchemaP {
    BLAZING_INTERFACE(NodeTableSchemaP);

public:
    virtual const BlazingTableSchema & blazingTableSchema() const noexcept = 0;

    virtual flatbuffers::Offset<NodeTable> CreateFlatBufferItem(
        flatbuffers::FlatBufferBuilder & flatBufferBuilder) const noexcept = 0;
};

class DeserializedNodeTableSchemaP : public NodeTableSchemaP {
    BLAZING_CONCRETE(DeserializedNodeTableSchemaP);

public:
    explicit DeserializedNodeTableSchemaP(
        const BlazingTableSchema & blazingTableSchema)
        : blazingTableSchema_{blazingTableSchema} {}

    const BlazingTableSchema & blazingTableSchema() const noexcept final {
        return blazingTableSchema_;
    }

    flatbuffers::Offset<NodeTable> CreateFlatBufferItem(
        flatbuffers::FlatBufferBuilder & flatBufferBuilder) const
        noexcept final {
        // Be carefull with a leak here. User does not ensure the existence of
        // the blazing table schema
        flatbuffers::Offset<BlazingTable> blazingTable =
            BlazingTableSchema::Serialize(flatBufferBuilder,
                                          blazingTableSchema_);
        return CreateNodeTable(flatBufferBuilder, blazingTable);
    }

private:
    const BlazingTableSchema & blazingTableSchema_;
};

class SerializedNodeTableSchemaP : public NodeTableSchemaP {
    BLAZING_CONCRETE(SerializedNodeTableSchemaP);

public:
    explicit SerializedNodeTableSchemaP(const NodeTable * nodeTable)
        : nodeTable_{nodeTable} {
        // Maybe here we have a unnecessary copy
        BlazingTableSchema::Deserialize(nodeTable->gdf(), &blazingTableSchema_);
    }

    const BlazingTableSchema & blazingTableSchema() const noexcept final {
        return blazingTableSchema_;
    }

    flatbuffers::Offset<NodeTable> CreateFlatBufferItem(
        flatbuffers::FlatBufferBuilder & flatBufferBuilder) const
        noexcept final {
        // Maybe here we have a unnecessary copy
        flatbuffers::Offset<BlazingTable> blazingTable =
            BlazingTableSchema::Serialize(flatBufferBuilder,
                                          blazingTableSchema_);
        return CreateNodeTable(flatBufferBuilder, blazingTable);
    }

private:
    const NodeTable *  nodeTable_;
    BlazingTableSchema blazingTableSchema_;
};

NodeTableSchema::NodeTableSchema(const BlazingTableSchema & blazingTableSchema)
    : nodeTableSchemaP_{
          std::make_shared<DeserializedNodeTableSchemaP>(blazingTableSchema)} {}

NodeTableSchema::NodeTableSchema(const NodeTable * nodeTable)
    : nodeTableSchemaP_{
          std::make_shared<SerializedNodeTableSchemaP>(nodeTable)} {}

const BlazingTableSchema & NodeTableSchema::blazingTable() const noexcept {
    return nodeTableSchemaP_->blazingTableSchema();
}

flatbuffers::Offset<NodeTable> NodeTableSchema::CreateFlatBufferItem(
    flatbuffers::FlatBufferBuilder & flatBufferBuilder) const noexcept {
    return nodeTableSchemaP_->CreateFlatBufferItem(flatBufferBuilder);
}

//   Node tables schema

class NodeTablesSchemaP {
    BLAZING_INTERFACE(NodeTablesSchemaP);

public:
    virtual bool IsEmpty() const noexcept = 0;

    virtual const NodeTableSchema Get(const std::size_t index) const
        noexcept = 0;

    virtual void Push(const NodeTableSchema & nodeTableSchema) noexcept = 0;

    virtual flatbuffers::Offset<
        flatbuffers::Vector<flatbuffers::Offset<NodeTable>>>
    CreateFlatBufferVector(
        flatbuffers::FlatBufferBuilder & flatBufferBuilder) const noexcept = 0;
};

class DeserializedNodeTablesSchemaP : public NodeTablesSchemaP {
    BLAZING_CONCRETE(DeserializedNodeTablesSchemaP);

public:
    explicit DeserializedNodeTablesSchemaP() = default;

    bool IsEmpty() const noexcept final { return nodeTableSchemas_.empty(); }

    const NodeTableSchema Get(const std::size_t index) const noexcept final {
        try {
            return *nodeTableSchemas_.at(index);
        } catch (const std::out_of_range &) {
            BLAZING_ABORT("You shouldn't be here");
        }
    }

    void Push(const NodeTableSchema & nodeTableSchema) noexcept final {
        nodeTableSchemas_.push_back(&nodeTableSchema);
    }

    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NodeTable>>>
    CreateFlatBufferVector(flatbuffers::FlatBufferBuilder & flatBufferBuilder)
        const noexcept final {
        std::vector<flatbuffers::Offset<NodeTable>> nodeTables;
        nodeTables.reserve(nodeTableSchemas_.size());
        std::transform(
            nodeTableSchemas_.cbegin(),
            nodeTableSchemas_.cend(),
            std::back_inserter(nodeTables),
            [&flatBufferBuilder](const NodeTableSchema * nodeTableSchema) {
                return nodeTableSchema->CreateFlatBufferItem(flatBufferBuilder);
            });
        return flatBufferBuilder.CreateVector(nodeTables.data(),
                                              nodeTables.size());
    }

private:
    std::vector<const NodeTableSchema *> nodeTableSchemas_;
};

class SerializedNodeTablesSchemaP : public NodeTablesSchemaP {
    BLAZING_CONCRETE(SerializedNodeTablesSchemaP);

public:
    explicit SerializedNodeTablesSchemaP(
        const flatbuffers::Vector<flatbuffers::Offset<NodeTable>> * nodeTables)
        : nodeTables_{nodeTables} {}

    bool IsEmpty() const noexcept final { return !nodeTables_->Length(); }

    const NodeTableSchema Get(const std::size_t index) const noexcept final {
        const NodeTable * nodeTable =
            nodeTables_->Get(static_cast<flatbuffers::uoffset_t>(index));
        return NodeTableSchema(nodeTable);
    }

    void Push(const NodeTableSchema & /*nodeTableSchema*/) noexcept final {
        BLAZING_ABORT("You shouldn't be here");
    }

    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NodeTable>>>
    CreateFlatBufferVector(flatbuffers::FlatBufferBuilder & flatBufferBuilder)
        const noexcept final {
        return flatBufferBuilder.CreateVector(nodeTables_->data(),
                                              nodeTables_->Length());
    }

private:
    const flatbuffers::Vector<flatbuffers::Offset<NodeTable>> * nodeTables_;
};

// This should not exist. Don't construct without node tables. We need this
// class beacuse IMessage's have a default constructor.
class DefaultNodeTablesSchemaP : public NodeTablesSchemaP {
public:
    bool IsEmpty() const noexcept final { return true; }

    const NodeTableSchema BLAZING_NORETURN
                          Get(const std::size_t /*index*/) const noexcept final {
        BLAZING_ABORT("You shouldn't be here");
    }

    void BLAZING_NORETURN
         Push(const NodeTableSchema & /*nodeTableSchema*/) noexcept final {
        BLAZING_ABORT("You shouldn't be here");
    }

    flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NodeTable>>>
    CreateFlatBufferVector(flatbuffers::FlatBufferBuilder & flatBufferBuilder)
        const noexcept final {
        return flatBufferBuilder.CreateVector<NodeTable>(nullptr, 0);
    }
};

NodeTablesSchema::NodeTablesSchema()
    : nodeTablesSchemaP_{std::make_unique<DeserializedNodeTablesSchemaP>()} {}

NodeTablesSchema::NodeTablesSchema(
    const flatbuffers::Vector<flatbuffers::Offset<NodeTable>> * nodeTables)
    : nodeTablesSchemaP_{
          std::make_unique<SerializedNodeTablesSchemaP>(nodeTables)} {}

bool NodeTablesSchema::IsEmpty() const noexcept {
    return nodeTablesSchemaP_->IsEmpty();
}

const NodeTableSchema NodeTablesSchema::
                      operator[](const std::size_t index) const noexcept {
    return nodeTablesSchemaP_->Get(index);
}

void NodeTablesSchema::Push(const NodeTableSchema & nodeTableSchema) noexcept {
    nodeTablesSchemaP_->Push(nodeTableSchema);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<NodeTable>>>
NodeTablesSchema::CreateFlatBufferVector(
    flatbuffers::FlatBufferBuilder & flatBufferBuilder) const noexcept {
    return nodeTablesSchemaP_->CreateFlatBufferVector(flatBufferBuilder);
}

}  // namespace orchestrator
}  // namespace protocol
}  // namespace blazingdb
