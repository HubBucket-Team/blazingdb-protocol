#pragma once

#include <string>
#include <functional>
#include <typeinfo>    

#include <blazingdb/protocol/api.h>
#include <iostream>
#include "flatbuffers/flatbuffers.h"
#include "generated/all_generated.h"

namespace libgdf {
  typedef size_t gdf_size_type;
  typedef gdf_size_type gdf_index_type;
  typedef unsigned char gdf_valid_type;
  typedef	long	gdf_date64;
  typedef	int		gdf_date32;
  typedef	int		gdf_category;

/* --------------------------------------------------------------------------*/
  /**
  * @Synopsis  These enums indicate the possible data types for a gdf_column
  */
/* ----------------------------------------------------------------------------*/
  typedef enum {
    GDF_invalid=0,
    GDF_INT8,
    GDF_INT16,
    GDF_INT32,
    GDF_INT64,
    GDF_FLOAT32,
    GDF_FLOAT64,
    GDF_DATE32,   	/**< int32_t days since the UNIX epoch */
    GDF_DATE64,   	/**< int64_t milliseconds since the UNIX epoch */
    GDF_TIMESTAMP,	/**< Exact timestamp encoded with int64 since UNIX epoch (Default unit millisecond) */
    GDF_CATEGORY,
    GDF_STRING,
    N_GDF_TYPES, 	/* additional types should go BEFORE N_GDF_TYPES */
  } gdf_dtype;


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  These are all possible gdf error codes that can be returned from
 * a libgdf function. ANY NEW ERROR CODE MUST ALSO BE ADDED TO `gdf_error_get_name`
 * AS WELL
 */
/* ----------------------------------------------------------------------------*/
  typedef enum {
    GDF_SUCCESS=0,
    GDF_CUDA_ERROR,                   /**< Error occured in a CUDA call */
    GDF_UNSUPPORTED_DTYPE,            /**< The datatype of the gdf_column is unsupported */
    GDF_COLUMN_SIZE_MISMATCH,         /**< Two columns that should be the same size aren't the same size*/
    GDF_COLUMN_SIZE_TOO_BIG,          /**< Size of column is larger than the max supported size */
    GDF_DATASET_EMPTY,                /**< Input dataset is either null or has size 0 when it shouldn't */
    GDF_VALIDITY_MISSING,             /**< gdf_column's validity bitmask is null */
    GDF_VALIDITY_UNSUPPORTED,         /**< The requested gdf operation does not support validity bitmask handling, and one of the input columns has the valid bits enabled */
    GDF_INVALID_API_CALL,             /**< The arguments passed into the function were invalid */
    GDF_JOIN_DTYPE_MISMATCH,          /**< Datatype mismatch between corresponding columns in  left/right tables in the Join function */
    GDF_JOIN_TOO_MANY_COLUMNS,        /**< Too many columns were passed in for the requested join operation*/
    GDF_DTYPE_MISMATCH,               /**< Type mismatch between columns that should be the same type */
    GDF_UNSUPPORTED_METHOD,           /**< The method requested to perform an operation was invalid or unsupported (e.g., hash vs. sort)*/
    GDF_INVALID_AGGREGATOR,           /**< Invalid aggregator was specified for a groupby*/
    GDF_INVALID_HASH_FUNCTION,        /**< Invalid hash function was selected */
    GDF_PARTITION_DTYPE_MISMATCH,     /**< Datatype mismatch between columns of input/output in the hash partition function */
    GDF_HASH_TABLE_INSERT_FAILURE,    /**< Failed to insert to hash table, likely because its full */
    GDF_UNSUPPORTED_JOIN_TYPE,        /**< The type of join requested is unsupported */
    GDF_UNDEFINED_NVTX_COLOR,         /**< The requested color used to define an NVTX range is not defined */
    GDF_NULL_NVTX_NAME,               /**< The requested name for an NVTX range cannot be nullptr */
    GDF_C_ERROR,				         	    /**< C error not related to CUDA */
    GDF_FILE_ERROR,   				        /**< error processing sepcified file */
  } gdf_error;

  typedef enum {
    GDF_HASH_MURMUR3=0, /**< Murmur3 hash function */
    GDF_HASH_IDENTITY,  /**< Identity hash function that simply returns the key to be hashed */
  } gdf_hash_func;

  typedef enum {
    TIME_UNIT_NONE=0, // default (undefined)
    TIME_UNIT_s,   // second
    TIME_UNIT_ms,  // millisecond
    TIME_UNIT_us,  // microsecond
    TIME_UNIT_ns   // nanosecond
  } gdf_time_unit;

  typedef struct {
    gdf_time_unit time_unit;
    // here we can also hold info for decimal datatype or any other datatype that requires additional information
  } gdf_dtype_extra_info;

  typedef struct gdf_column_{
    void *data;                       /**< Pointer to the columns data */
    gdf_valid_type *valid;            /**< Pointer to the columns validity bit mask where the 'i'th bit indicates if the 'i'th row is NULL */
    gdf_size_type size;               /**< Number of data elements in the columns data buffer*/
    gdf_dtype dtype;                  /**< The datatype of the column's data */
    gdf_size_type null_count;         /**< The number of NULL values in the column's data */
    gdf_dtype_extra_info dtype_info;
  } gdf_column;

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  These enums indicate which method is to be used for an operation.
 * For example, it is used to select between the hash-based vs. sort-based implementations
 * of the Join operation.
 */
/* ----------------------------------------------------------------------------*/
  typedef enum {
    GDF_SORT = 0,   /**< Indicates that the sort-based implementation of the function will be used */
    GDF_HASH,       /**< Indicates that the hash-based implementation of the function will be used */
    N_GDF_METHODS,  /* additional methods should go BEFORE N_GDF_METHODS */
  } gdf_method;

  typedef enum {
    GDF_QUANT_LINEAR =0,
    GDF_QUANT_LOWER,
    GDF_QUANT_HIGHER,
    GDF_QUANT_MIDPOINT,
    GDF_QUANT_NEAREST,
    N_GDF_QUANT_METHODS,
  } gdf_quantile_method;


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis These enums indicate the supported aggregation operations that can be
 * performed on a set of aggregation columns as part of a GroupBy operation
 */
/* ----------------------------------------------------------------------------*/
  typedef enum {
    GDF_SUM = 0,        /**< Computes the sum of all values in the aggregation column*/
    GDF_MIN,            /**< Computes minimum value in the aggregation column */
    GDF_MAX,            /**< Computes maximum value in the aggregation column */
    GDF_AVG,            /**< Computes arithmetic mean of all values in the aggregation column */
    GDF_COUNT,          /**< Computes histogram of the occurance of each key in the GroupBy Columns */
    GDF_COUNT_DISTINCT, /**< Counts the number of distinct keys in the GroupBy columns */
    N_GDF_AGG_OPS,      /**< The total number of aggregation operations. ALL NEW OPERATIONS SHOULD BE ADDED ABOVE THIS LINE*/
  } gdf_agg_op;


/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Colors for use with NVTX ranges.
 *
 * These enumerations are the available pre-defined colors for use with
 * user-defined NVTX ranges.
 */
/* ----------------------------------------------------------------------------*/
  typedef enum {
    GDF_GREEN = 0,
    GDF_BLUE,
    GDF_YELLOW,
    GDF_PURPLE,
    GDF_CYAN,
    GDF_RED,
    GDF_WHITE,
    GDF_DARK_GREEN,
    GDF_ORANGE,
    GDF_NUM_COLORS, /** Add new colors above this line */
  } gdf_color;

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  This struct holds various information about how an operation should be
 * performed as well as additional information about the input data.
 */
/* ----------------------------------------------------------------------------*/
  typedef struct gdf_context_{
    int flag_sorted;        /**< Indicates if the input data is sorted. 0 = No, 1 = yes */
    gdf_method flag_method; /**< The method to be used for the operation (e.g., sort vs hash) */
    int flag_distinct;      /**< for COUNT: DISTINCT = 1, else = 0 */
    int flag_sort_result;   /**< When method is GDF_HASH, 0 = result is not sorted, 1 = result is sorted */
    int flag_sort_inplace;  /**< 0 = No sort in place allowed, 1 = else */
  } gdf_context;

  struct _OpaqueIpcParser;
  typedef struct _OpaqueIpcParser gdf_ipc_parser_type;


  struct _OpaqueRadixsortPlan;
  typedef struct _OpaqueRadixsortPlan gdf_radixsort_plan_type;


  struct _OpaqueSegmentedRadixsortPlan;
  typedef struct _OpaqueSegmentedRadixsortPlan gdf_segmented_radixsort_plan_type;




  typedef enum{
    GDF_ORDER_ASC,
    GDF_ORDER_DESC
  } order_by_type;

  typedef enum{
    GDF_EQUALS,
    GDF_NOT_EQUALS,
    GDF_LESS_THAN,
    GDF_LESS_THAN_OR_EQUALS,
    GDF_GREATER_THAN,
    GDF_GREATER_THAN_OR_EQUALS
  } gdf_comparison_operator;

  typedef enum{
    GDF_WINDOW_RANGE,
    GDF_WINDOW_ROW
  } window_function_type;

  typedef enum{
    GDF_WINDOW_AVG,
    GDF_WINDOW_SUM,
    GDF_WINDOW_MAX,
    GDF_WINDOW_MIN,
    GDF_WINDOW_COUNT,
    GDF_WINDOW_STDDEV,
    GDF_WINDOW_VAR //variance
  } window_reduction_type;
}

namespace blazingdb {
namespace protocol {

class IMessage {
public:
  IMessage() = default;

  virtual ~IMessage() = default;

  virtual std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const = 0;

};


class ResponseMessage  : public IMessage {
public:  
  ResponseMessage (const uint8_t* buffer) 
      : IMessage()
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Response>(buffer);
      status_ = pointer->status();
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
  }

  ResponseMessage(Status status, std::shared_ptr<flatbuffers::DetachedBuffer>& buffer) : IMessage() {
      status_ = status;
      _copy_payload = buffer;

      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
  }

  ResponseMessage(Status status, IMessage& payload) : IMessage() {
    status_ = status;
    _copy_payload = payload.getBufferData();

    payloadBuffer = _copy_payload->data();
    payloadBufferSize = _copy_payload->size();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    flatbuffers::FlatBufferBuilder builder{0};

    auto payload_offset = builder.CreateVector(payloadBuffer, payloadBufferSize);
    auto root_offset = CreateResponse(builder, status_, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  Status getStatus() {
    return status_;
  } 
  const uint8_t* getPayloadBuffer() {
    return payloadBuffer;
  }
private:
    Status            status_;
    uint8_t*          payloadBuffer;
    size_t            payloadBufferSize;
    std::shared_ptr<flatbuffers::DetachedBuffer>  _copy_payload; 

 };

class ResponseErrorMessage : public IMessage {
public:  

  ResponseErrorMessage(const std::string& error) : IMessage(), error (error)
  {
  }
  
  ResponseErrorMessage (const uint8_t* buffer) : IMessage() {
    auto pointer = flatbuffers::GetRoot<blazingdb::protocol::ResponseError>(buffer);
    
    error = std::string{pointer->errors()->c_str()};
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override  {
    flatbuffers::FlatBufferBuilder builder{1024};
    auto string_offset = builder.CreateString(error);
    auto root_offset = CreateResponseError(builder, string_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  std::string getMessage () {
    return error;
  }
  
private:
  std::string error;
};

static inline const Header * GetHeaderPtr (const uint8_t* buffer) {
  return flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer)->header();
}

class RequestMessage : public IMessage {
public:  
  RequestMessage (const uint8_t* buffer) 
    : IMessage(), header{GetHeaderPtr(buffer)->messageType(),
                         GetHeaderPtr(buffer)->accessToken() } 
  {
      auto pointer = flatbuffers::GetRoot<blazingdb::protocol::Request>(buffer);
      payloadBuffer = (uint8_t*)pointer->payload()->data();
      payloadBufferSize = pointer->payload()->size();
      
  }
  RequestMessage(Header &&_header, IMessage& payload) 
      : IMessage(), header{_header} 
  {
      _copy_payload = payload.getBufferData(); 
      payloadBuffer = _copy_payload->data();
      payloadBufferSize = _copy_payload->size();
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData() const override {
    flatbuffers::FlatBufferBuilder builder{0};
    auto payload_offset = builder.CreateVector(payloadBuffer, payloadBufferSize);
    auto root_offset = CreateRequest(builder, &header, payload_offset);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

  Buffer getPayloadBuffer() {
    return Buffer{payloadBuffer, payloadBufferSize};
  }

  size_t getPayloadBufferSize() {
    return  payloadBufferSize;
  }

  int8_t  messageType() const { 
    return header.messageType();
  }

  uint64_t  accessToken() const {
    return header.accessToken();
  }


private:
    Header            header;
    const uint8_t*    payloadBuffer;
    size_t            payloadBufferSize;
    std::shared_ptr<flatbuffers::DetachedBuffer>  _copy_payload; 
};


template<typename T, typename SchemaType>
class TypedMessage : public IMessage {
public:
  TypedMessage(const T& val) : IMessage(), value_ (val){

  }

  using PointerToMethod = T (SchemaType::*)() const;

  TypedMessage (const uint8_t* buffer, PointerToMethod pmfn)
      : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<SchemaType>(buffer);
    value_ = (pointer->*pmfn)();
  }

  template<class CreateFunctionPtr>
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferDataUsing(CreateFunctionPtr &&create_function) const  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto root_offset = create_function(builder, value_);
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }

protected:
  T value_;
};

template<typename SchemaType>
class StringTypeMessage : public IMessage {
public: 
  StringTypeMessage(const std::string& string) : IMessage(), string_value (string){

  }
  
  using PointerToMethod = const flatbuffers::String* (SchemaType::*)() const;

  StringTypeMessage (const uint8_t* buffer, PointerToMethod pmfn)
    : IMessage()
  {
    auto pointer = flatbuffers::GetRoot<SchemaType>(buffer);
    auto string_buffer = (pointer->*pmfn)();
    string_value = std::string {string_buffer->c_str()};
  }

  template<class CreateFunctionPtr>
  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferDataUsing(CreateFunctionPtr &&create_function) const  {
    flatbuffers::FlatBufferBuilder builder{0};
    auto root_offset = create_function(builder, string_value.c_str());
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
  
protected:
  std::string string_value;
 };


class  ZeroMessage : public IMessage {
public:

  ZeroMessage()
      : IMessage()
  {
  }

  std::shared_ptr<flatbuffers::DetachedBuffer> getBufferData( ) const override  {
    flatbuffers::FlatBufferBuilder builder{};
    auto root_offset = builder.CreateString("");
    builder.Finish(root_offset);
    return std::make_shared<flatbuffers::DetachedBuffer>(builder.Release());
  }
};



auto MakeRequest(int8_t message_type, uint64_t sessionToken, IMessage&& payload) -> std::shared_ptr<flatbuffers::DetachedBuffer>{
  RequestMessage request{ Header{message_type, sessionToken}, payload}; 
  auto bufferedData = request.getBufferData();
  return bufferedData;
}
auto MakeRequest(int8_t message_type, uint64_t sessionToken, IMessage& payload) -> std::shared_ptr<flatbuffers::DetachedBuffer>{
  RequestMessage request{ Header{message_type, sessionToken}, payload}; 
  auto bufferedData = request.getBufferData();
  return bufferedData;
}
 

template <typename ResponseType>
ResponseType MakeResponse (Buffer &responseBuffer) {
  ResponseMessage response{responseBuffer.data()};
  if (response.getStatus() == Status_Error) {
    ResponseErrorMessage errorMessage{response.getPayloadBuffer()};
    throw std::runtime_error(errorMessage.getMessage());
  }
  ResponseType responsePayload(response.getPayloadBuffer());
  return responsePayload;
}


struct BlazingTableDTO {
  std::string name;
  std::vector<::libgdf::gdf_column> columns;
  std::vector<std::string> columnNames;
};

struct TableGroupDTO {
  std::vector<BlazingTableDTO> tables;
  std::string name;
};

//@todo using cuda_ipc_mem_handler
static flatbuffers::Offset<flatbuffers::Vector<int8_t>> BuildCudaIpcMemHandler (flatbuffers::FlatBufferBuilder &builder, void *data) {
  flatbuffers::Offset<flatbuffers::Vector<int8_t>> offsets;
  //@todo using cuda_ipc_mem_handler
  int8_t bytes[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  return builder.CreateVector(bytes, 10);
}

//@todo using cuda_ipc_mem_handler
static void* CudaIpcMemHandlerFrom (const gdf::cudaIpcMemHandle_t *handler) {
  void * response = nullptr;

  return response;
}


static TableGroupDTO TableGroupDTOFrom(const blazingdb::protocol::TableGroup * tableGroup) {
  std::string name = std::string{tableGroup->name()->c_str()};
  std::vector<BlazingTableDTO> tables;

  auto rawTables = tableGroup->tables();
  for (const auto& table : *rawTables) {
    std::vector<::libgdf::gdf_column>  columns;
    std::vector<std::string>  columnNames;

    for (const auto& c : *table->columns()){
      ::libgdf::gdf_column column = {
          .data = CudaIpcMemHandlerFrom(c->data()),
          .valid = (unsigned char *)CudaIpcMemHandlerFrom(c->valid()),
          .size = c->size(),
          .dtype = (libgdf::gdf_dtype)c->dtype(),
          .null_count = c->null_count(),
          .dtype_info = libgdf::gdf_dtype_extra_info {
             .time_unit = (libgdf::gdf_time_unit) c->dtype_info()->time_unit()
          },
      };
      columns.push_back(column);
    }

    tables.push_back(BlazingTableDTO{
        .name = std::string{table->name()->c_str()},
        .columns = columns,
        .columnNames = columnNames,
    });
  }

  return TableGroupDTO {
    .tables = tables,
    .name = name,
  };
}

static flatbuffers::Offset<TableGroup> BuildTableGroup(flatbuffers::FlatBufferBuilder &builder,
                                                       const TableGroupDTO &tableGroup) {
  auto tableNameOffset = builder.CreateString(tableGroup.name);
  std::vector<flatbuffers::Offset<BlazingTable>> tablesOffset;

  auto _createColumns = [] (flatbuffers::FlatBufferBuilder &builder, std::vector<::libgdf::gdf_column> &columns) -> std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> {
    std::vector<flatbuffers::Offset<gdf::gdf_column_handler>> offsets;
    for (auto & c: columns) {
      auto dtype_extra_info = gdf::Creategdf_dtype_extra_info (builder, (gdf::gdf_time_unit)c.dtype_info.time_unit );
      auto data_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler (builder, c.data) );
      auto valid_offset = gdf::CreatecudaIpcMemHandle_t(builder, BuildCudaIpcMemHandler(builder, c.valid) );
      auto column_offset = ::blazingdb::protocol::gdf::Creategdf_column_handler(builder, data_offset, valid_offset, c.size, (gdf::gdf_dtype)c.dtype, dtype_extra_info);
      offsets.push_back(column_offset);
    }
    return offsets;
  };
  auto _createColumnNames  = [] (flatbuffers::FlatBufferBuilder &builder, std::vector<std::string> &columnNames) -> std::vector<flatbuffers::Offset<flatbuffers::String>> {
    std::vector<flatbuffers::Offset<flatbuffers::String>> offsets;
    for (auto & name: columnNames) {
      offsets.push_back( builder.CreateString(name.data()));
    }
    return offsets;
  };
  for (auto table : tableGroup.tables) {
    auto columns = _createColumns(builder, table.columns);
    auto columnNames = _createColumnNames(builder, table.columnNames);
    tablesOffset.push_back( CreateBlazingTable(builder, builder.CreateString(table.name), builder.CreateVector(columns), builder.CreateVector(columnNames)));
  }

  auto tables = builder.CreateVector(tablesOffset);
  return CreateTableGroup(builder, tables, tableNameOffset);
}

}
}