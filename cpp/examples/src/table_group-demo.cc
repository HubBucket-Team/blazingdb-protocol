#include <tuple>

#include <gdf/gdf.h>
using namespace gdf;

#include "gdf/library/table.h"
#include "gdf/library/column.h"

#include <iostream>
#include <string>
#include <vector>

struct Item {
  std::string query;
  std::string logicalPlan;

  //  [VTable] : tableGroup 
    // std::vector<std::string> dataTypes;
    // std::vector<std::string> resultTypes;
    // std::vector<std::vector<std::string> > data;

  //  VTable: result
    // std::vector<std::vector<std::string> > result;
};

// std::vector<Item> inputSet{
// Item{"select * from main.emps",
//  "LogicalProject(id=[$0], age=[$1])\n  EnumerableTableScan(table=[[main, emps]])", 
// {"GDF_INT8","GDF_INT8"}, {"GDF_INT8","GDF_INT8"}, 
// {{"1","2","3","4","5","6","7","8","9","10"},{"10","20","30","40","50","60","70","80","90","100"}},
// {{"1","2","3","4","5","6","7","8","9","10"},{"10","20","30","40","50","60","70","80","90","100"}}},


int main(){ 

  gdf::library::Table t =
      gdf::library::TableBuilder{
          "emps",
          {
            {"x", [](const std::size_t) -> gdf::library::Ret<GDF_FLOAT64> { return .1; }},
            {"y", [](const std::size_t i) -> gdf::library::Ret<GDF_UINT64> { return i; }},
          }
        }
        .build(10);

  t.print(std::cout);

 
  using VTableBuilder = gdf::library::TableRowBuilder<int8_t, double, int32_t, int64_t>;
  using DataTuple = VTableBuilder::DataTuple;

  gdf::library::Table table = 
      VTableBuilder {
        .name = "emps",
        .headers = {"Id", "Weight", "Age", "Name"},
        .rows = {
          DataTuple{'a', 180.2, 40, 100L},
          DataTuple{'b', 175.3, 38, 200L},
          DataTuple{'c', 140.3, 27, 300L},
        },
      }
      .build();

  table.print(std::cout);
  
  
  return 0;
}