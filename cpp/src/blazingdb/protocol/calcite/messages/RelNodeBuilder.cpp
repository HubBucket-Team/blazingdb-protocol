#include "RelNodeBuilder.hpp"
#include "LogicalPlan.h"
#include "../../message/generated/all_generated.h"

namespace blazingdb {
namespace protocol {
namespace calcite {
namespace messages {

namespace LogicalPlan = blazingdb::protocol::dto;
namespace Calcite = com::blazingdb::protocol::calcite::plan::messages;

class RelNodeBuilder::RelNodeBuilderImpl {
public:
    RelNodeBuilderImpl(const Buffer<std::uint8_t> &buffer) : buffer_{buffer} {}
    inline virtual ~RelNodeBuilderImpl() = default;

    LogicalPlan::RelNodePtr Build() const;

private:
    void buildRelNodeDTO(LogicalPlan::RelNodePtr& parent, const Calcite::RelNode* input) const;
    void createRelNodeDTO(LogicalPlan::RelNodePtr& output, const Calcite::RelNode* input) const;

private:
    const Buffer<std::uint8_t> buffer_;
};

RelNodeBuilder::RelNodeBuilder(const Buffer<std::uint8_t> &buffer)
    : impl_{new RelNodeBuilderImpl{buffer}} {}

RelNodeBuilder::~RelNodeBuilder() = default;

LogicalPlan::RelNodePtr RelNodeBuilder::Build() const { return impl_->Build(); }

LogicalPlan::RelNodePtr RelNodeBuilder::RelNodeBuilderImpl::Build() const {
    LogicalPlan::RelNodePtr output {nullptr};
    const Calcite::RelNode* relNode {nullptr};

    relNode = flatbuffers::GetRoot<Calcite::RelNode>(buffer_.data());
    buildRelNodeDTO(output, relNode);

    return output;
}

void RelNodeBuilder::RelNodeBuilderImpl::buildRelNodeDTO(LogicalPlan::RelNodePtr& output, const Calcite::RelNode* input) const {
    createRelNodeDTO(output, input);

    auto* inputs = input->inputs();
    if (inputs == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < inputs->size(); ++i) {
        LogicalPlan::RelNodePtr node;
        createRelNodeDTO(node, inputs->Get(i));
        output->addInput(node);
        buildRelNodeDTO(node, inputs->Get(i));
    }
}

void RelNodeBuilder::RelNodeBuilderImpl::createRelNodeDTO(LogicalPlan::RelNodePtr& output, const Calcite::RelNode* input) const {
    switch (input->type()) {
        case Calcite::RelNodeType_LogicalProject: {
            output = LogicalPlan::RelFactory::createLogicalProject();
            break;
        }
        case Calcite::RelNodeType_LogicalFilter: {
            output = LogicalPlan::RelFactory::createLogicalFilter();
            break;
        }
        case Calcite::RelNodeType_TableScan: {
            output = LogicalPlan::RelFactory::createTableScan();
            break;
        }
        case Calcite::RelNodeType_LogicalAggregate: {
            output = LogicalPlan::RelFactory::createLogicalAggregate();
            break;
        }
        case Calcite::RelNodeType_LogicalUnion: {
            output = LogicalPlan::RelFactory::createLogicalUnion();
            break;
        }
    }
}

}  // namespace messages
}  // namespace calcite
}  // namespace protocol
}  // namespace blazingdb
