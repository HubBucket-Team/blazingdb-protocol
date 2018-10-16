#include <iostream>

#include <blazingdb/protocol/api.h>

#include "ral-client.cuh"

#include "../gdf/GDFColumn.cuh"

using namespace blazingdb::protocol;
 
int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/ral.socket");
  interpreter::InterpreterClient client{connection};

  std::string logicalPlan = "\
    LogicalProject(EXPR$0=[>($0, 5)])\n\
      EnumerableTableScan(table=[[hr, emps]])";

  libgdf::gdf_column_cpp one;
  libgdf::create_sample_gdf_column(one); 
  libgdf::print_column(one.get_gdf_column());

  try {
    auto tableGroup = ::blazingdb::protocol::TableGroupDTO{
      .tables = {
          BlazingTableDTO {
              .name="user",
              .columns = {
                  ::libgdf::gdf_column {
                        .data = one.data(),
                        .valid = one.valid(),
                        .size = one.size(),
                        .dtype = (libgdf::gdf_dtype)one.dtype(),
                        .null_count = one.null_count(),
                        .dtype_info = libgdf::gdf_dtype_extra_info {
                        .time_unit = (libgdf::gdf_time_unit)0,
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
    libgdf::print_column( &resultSet[0] );

  } catch (std::runtime_error &error) {
    std::cout << error.what() << std::endl;
  }
  return 0;
}