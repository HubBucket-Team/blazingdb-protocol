#include "table.h"

class TableGroup {
public:
  TableGroup(std::initializer_list<Table> tables) {
    for (Table table : tables) { tables_.push_back(table); }
  }

  TableGroup(std::initializer_list<std::initializer_list<gdf_dtype> > dtypess) {
  }

  BlazingFrame ToBlazingFrame() const;

  const Table &operator[](const std::size_t i) const { return tables_[i]; }

private:
  std::vector<Table> tables_;
};
 

BlazingFrame TableGroup::ToBlazingFrame() const {
  BlazingFrame frame;
  return frame;
}