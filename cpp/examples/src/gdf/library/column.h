#pragma once 

#include <cassert>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>
#include <iostream>
#include <type_traits>

#include <gdf/gdf.h>

#include "types.h"
#include "vector.h"
#include "gdf_column.h"
#include "any.h"

namespace gdf {
namespace library {


template <gdf_dtype DTYPE>
struct DTypeTraits {};

#define DTYPE_FACTORY(DTYPE, T)                                                \
  template <>                                                                  \
  struct DTypeTraits<GDF_##DTYPE> {                                            \
    typedef T value_type;                                                      \
  }

DTYPE_FACTORY(INT8, std::int8_t);
DTYPE_FACTORY(INT16, std::int16_t);
DTYPE_FACTORY(INT32, std::int32_t);
DTYPE_FACTORY(INT64, std::int64_t);
DTYPE_FACTORY(UINT8, std::uint8_t);
DTYPE_FACTORY(UINT16, std::uint16_t);
DTYPE_FACTORY(UINT32, std::uint32_t);
DTYPE_FACTORY(UINT64, std::uint64_t);
DTYPE_FACTORY(FLOAT32, float);
DTYPE_FACTORY(FLOAT64, double);
DTYPE_FACTORY(DATE32, std::int32_t);
DTYPE_FACTORY(DATE64, std::int64_t);
DTYPE_FACTORY(TIMESTAMP, std::int64_t);

#undef DTYPE_FACTORY


class Column {
public:
  Column(const std::string &name)
    : name_{name}
    {}

  ~Column();

  virtual GdfColumn ToGdfColumnCpp() const = 0;
  virtual const void *   get_values() const     = 0;
  virtual size_t   size() const   = 0;

  virtual size_t   print(std::ostream &stream) const   = 0;
  virtual std::string get_as_str(int index) const = 0;

  
  template <gdf_dtype DType>
  typename DTypeTraits<DType>::value_type get(const std::size_t i) const {
    return (*reinterpret_cast<
            const std::basic_string<typename DTypeTraits<DType>::value_type> *>(
      get_values()))[i];
  }
  
  std::string   name() const { return name_; }

protected:
  static GdfColumn Create(const gdf_dtype   dtype,
                               const std::size_t length,
                               const void *      data,
                               const std::size_t size);

protected:
  const std::string name_; 
};


Column::~Column() {}

GdfColumn Column::Create(const gdf_dtype   dtype,
                              const std::size_t length,
                              const void *      data,
                              const std::size_t size) {
  GdfColumn column_cpp;
  column_cpp.create_gdf_column(dtype, length, const_cast<void *>(data), size);
  return column_cpp;
}


template <gdf_dtype DType>
class TypedColumn : public Column {
public:
  using value_type = typename DTypeTraits<DType>::value_type;
  using Callback   = std::function<value_type(const std::size_t)>;

  TypedColumn(const std::string &name) : Column(name) {}

  void
  range(const std::size_t begin, const std::size_t end, Callback callback) {
    assert(end > begin);
    values_.reserve(end - begin);
    for (std::size_t i = begin; i < end; i++) {
      values_.push_back(callback(i));
    }
  }

  void range(const std::size_t end, Callback callback) {
    range(0, end, callback);
  }

  void FillData (std::vector< value_type > values) {
    for (std::size_t i = 0; i < values.size(); i++) {
      values_.push_back(values.at(i));
    }
  }
  GdfColumn ToGdfColumnCpp() const final {
    return Create(DType, this->size(), values_.data(), sizeof(value_type));
  }
  
  size_t  size() const final {
    return values_.size();
  }

  size_t   print(std::ostream &stream) const   final  {
    for (std::size_t i = 0; i < values_.size(); i++) {
      stream << values_.at(i) << " | " ;
    }
  }

  std::string get_as_str(int index) const  final  {
    std::ostringstream out;
    if (std::is_floating_point<value_type>::value) { 
      out.precision(1);
    }
    out << std::fixed << values_.at(index);
    return out.str(); 
  }

  const void *get_values() const final { return &values_; }

private:
  std::basic_string<value_type> values_;
};

template <class T>
class RangeTraits : public RangeTraits<decltype(&T::operator())> {};

template <class C, class R, class... A>
class RangeTraits<R (C::*)(A...) const> {
public:
  typedef R r_type;
};

template <gdf_dtype DType>
class Ret {
public:
  static constexpr gdf_dtype dtype = DType;

  using value_type = typename DTypeTraits<DType>::value_type;

  template <class T>
  Ret(const T value) : value_{value} {}

  operator value_type() const { return value_; }

private:
  const value_type value_;
};


class ColumnBuilder {
public:
  template <class C>
  ColumnBuilder(const std::string &name, C callback) {
    typedef RangeTraits<decltype(callback)> a;
    auto *column = new TypedColumn<a::r_type::dtype>(name);
    column->range(100, callback);
    column_ = column;
  } 
  Column *column_ptr() const { return column_; }

private:
  Column *column_;
};


class ColumnBuilderTyped {
public: 
  template <class Type>
  ColumnBuilderTyped(const std::string &name, std::vector< Type > values) {

    auto *column = new TypedColumn< GdfDataType<Type>::Value >(name);
    column->FillData(values);
    column_ = column;
  }

  Column *column_ptr() const { return column_; }

private:
  Column *column_;
};


} // namespace library
} // namesapce gdf
