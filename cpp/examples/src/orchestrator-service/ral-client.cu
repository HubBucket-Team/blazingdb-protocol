#include <iostream>

#include <blazingdb/protocol/api.h>

#include "ral-client.h"

#include "gdf/gdf.h"
#include "gdf/container/gdf_vector.cuh"
#include "gdf/util/gdf_utils.cuh"

using namespace blazingdb::protocol;
 
int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/ral.socket");
  interpreter::InterpreterClient client{connection};

  std::string logicalPlan = "\
    LogicalProject(EXPR$0=[>($0, 5)])\n\
      EnumerableTableScan(table=[[hr, emps]])";

  ::gdf::container::GdfVector  one;
  ::gdf::util::create_sample_gdf_column(one); 
  ::gdf::util::print_gdf_column(one.get_gdf_column());

  try {
    auto tableGroup = ::blazingdb::protocol::TableGroupDTO{
      .tables = {
          BlazingTableDTO {
              .name="user",
              .columns = {
                  ::gdf_dto::gdf_column {
                        .data = ::gdf::util::BuildCudaIpcMemHandler(one.data()),
                        .valid = ::gdf::util::BuildCudaIpcMemHandler(one.valid()),
                        .size = one.size(),
                        .dtype = (gdf_dto::gdf_dtype)one.dtype(),
                        .null_count = one.null_count(),
                        .dtype_info = gdf_dto::gdf_dtype_extra_info {
                        .time_unit = (gdf_dto::gdf_time_unit)0,
                      },
                  },
              },
              .columnNames = {"id", "name"}
          }
      },
      .name = "alexdb",
    };
    auto resultToken = client.executePlan(logicalPlan, tableGroup, 123456L);
    std::cout << "executePlan:\n";
    std::cout << resultToken << std::endl;

    auto resultSet = client.getResult(resultToken, 123456L);
    std::cout << "get result:\n";
    ::gdf::util::DtoToGdfColumn(resultSet);


  } catch (std::runtime_error &error) {
    std::cout << error.what() << std::endl;
  }
  return 0;
}