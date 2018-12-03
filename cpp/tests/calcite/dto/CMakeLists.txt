
include(${CMAKE_SOURCE_DIR}/src/blazingdb/protocol/calcite/dto/CMakeLists.txt)

set(logical_plan_name "logical-plan-test")

set(logical_plan_source_files ${source_files})

add_executable(${logical_plan_name} ${logical_plan_source_files})

add_test(NAME ${logical_plan_name} COMMAND ${logical_plan_name})

target_sources(${logical_plan_name} PUBLIC LogicalPlanTest.cpp)

target_link_libraries(${logical_plan_name} gmock gmock_main GTest::GTest GTest::Main)

set_target_properties(${logical_plan_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

target_include_directories(${logical_plan_name} PUBLIC "${CMAKE_SOURCE_DIR}/src/blazingdb/protocol/calcite/dto")
