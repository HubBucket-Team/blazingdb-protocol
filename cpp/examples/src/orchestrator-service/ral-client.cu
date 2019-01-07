#include <iostream>

#include <blazingdb/protocol/api.h>

#include "ral-client.h"

#include "gdf/gdf.h"
#include "gdf/library/gdf_column.h"
#include "gdf/util/gdf_utils.cuh"

using namespace blazingdb::protocol;

int main() {
  blazingdb::protocol::UnixSocketConnection connection("/tmp/ral.socket");
  interpreter::InterpreterClient client{connection};

  std::string logicalPlan = "\
    LogicalProject(EXPR$0=[>($0, 5)])\n\
      EnumerableTableScan(table=[[main, nation]])";

  ::gdf::library::GdfColumn  one;
  ::gdf::util::create_sample_gdf_column(one);
  ::gdf::util::print_gdf_column(one.get_gdf_column());

  ::gdf::library::GdfColumn  two;
  ::gdf::util::create_sample_gdf_column(two);
  ::gdf::util::print_gdf_column(two.get_gdf_column());

  ::gdf::library::GdfColumn  three;
  ::gdf::util::create_sample_gdf_column(three);
  ::gdf::util::print_gdf_column(three.get_gdf_column());


  try {
    auto tableGroup = ::blazingdb::protocol::TableGroupDTO{
      .tables = {
          BlazingTableDTO {
              .name="main.nation",
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
                  ::gdf_dto::gdf_column {
                        .data = ::gdf::util::BuildCudaIpcMemHandler(two.data()),
                        .valid = ::gdf::util::BuildCudaIpcMemHandler(two.valid()),
                        .size = two.size(),
                        .dtype = (gdf_dto::gdf_dtype)two.dtype(),
                        .null_count = two.null_count(),
                        .dtype_info = gdf_dto::gdf_dtype_extra_info {
                        .time_unit = (gdf_dto::gdf_time_unit)0,
                      },
                  },
                  ::gdf_dto::gdf_column {
                        .data = ::gdf::util::BuildCudaIpcMemHandler(three.data()),
                        .valid = ::gdf::util::BuildCudaIpcMemHandler(three.valid()),
                        .size = three.size(),
                        .dtype = (gdf_dto::gdf_dtype)three.dtype(),
                        .null_count = three.null_count(),
                        .dtype_info = gdf_dto::gdf_dtype_extra_info {
                        .time_unit = (gdf_dto::gdf_time_unit)0,
                      },
                  },
              },
              .columnNames = {"col1, col2, col3"}
          }
      },
      .name = "main",
    };
    auto resultBuffer = client.executePlan(logicalPlan, tableGroup, 123456L);
    auto resultToken = 1234L;
    auto resultSet = client.getResult(resultToken, 123456L);
    std::cout << "get result:\n";
    ::gdf::util::DtoToGdfColumn(resultSet);


  } catch (std::runtime_error &error) {
    std::cout << error.what() << std::endl;
  }
  return 0;
}