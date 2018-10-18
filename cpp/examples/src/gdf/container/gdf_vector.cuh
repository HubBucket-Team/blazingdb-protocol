#pragma once 

#include <map>
#include <mutex>

#include "gdf/gdf.h"

namespace gdf {
namespace container {
 

typedef std::pair<void*, gdf_valid_type*> rc_key_t; // std::pair<void* data, gdf_valid_type* valid>

class GDFRefCounter
{
	private:
		GDFRefCounter();

		static GDFRefCounter* Instance;

		std::mutex gc_mutex;

		std::map<rc_key_t, size_t> map; // std::map<key_ptr, ref_counter>

	public:
		void increment(gdf_column* col_ptr);

		void decrement(gdf_column* col_ptr);

		void register_column(gdf_column* col_ptr);

		void deregister_column(gdf_column* col_ptr);

		void free_if_deregistered(gdf_column* col_ptr);

		size_t get_map_size();

		static GDFRefCounter* getInstance();
};

GDFRefCounter* GDFRefCounter::Instance=0;

void GDFRefCounter::register_column(gdf_column* col_ptr){

    if(col_ptr != nullptr){
        std::lock_guard<std::mutex> lock(gc_mutex);
        rc_key_t map_key = {col_ptr->data, col_ptr->valid};
        
        if(map.find(map_key) == map.end()){
            map[map_key]=1;
        }
    }
}

void GDFRefCounter::deregister_column(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key) != map.end()){
        map[map_key]=0; //deregistering
    }
}

void GDFRefCounter::free_if_deregistered(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]==0){
            map.erase(map_key);
            cudaFree(map_key.first); //data
            cudaFree(map_key.second); //valid
        }
    }
}

void GDFRefCounter::increment(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]!=0){ //is already deregistered
            map[map_key]++;
        }
    }
}

void GDFRefCounter::decrement(gdf_column* col_ptr)
{
    std::lock_guard<std::mutex> lock(gc_mutex);
    rc_key_t map_key = {col_ptr->data, col_ptr->valid};

    if(map.find(map_key)!=map.end()){
        if(map[map_key]>0){
            map[map_key]--;

            if(map[map_key]==0){
                map.erase(map_key);
                cudaFree(map_key.first); //data
                cudaFree(map_key.second); //valid
            }
        }
    }
}

GDFRefCounter::GDFRefCounter()
{

}

// Testing purposes
size_t GDFRefCounter::get_map_size()
{
    return map.size();
}

GDFRefCounter* GDFRefCounter::getInstance()
{
    if(!Instance)
        Instance=new GDFRefCounter();
    return Instance;
}
 

class GdfVector
{
	private:
		gdf_column column;

	public:

    void* data();

    gdf_valid_type* valid();

    gdf_size_type size();

    gdf_dtype dtype();

    gdf_size_type null_count();

    gdf_dtype_extra_info dtype_info();

	void set_dtype(gdf_dtype dtype);

	GdfVector();

	GdfVector(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	GdfVector(const GdfVector& col);

	GdfVector(GdfVector& col);

	void operator=(const GdfVector& col);

	gdf_column* get_gdf_column();

	void create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value);

	void realloc_gdf_column(gdf_dtype type, size_t size, size_t width);

	gdf_error gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype);

	~GdfVector();
};


GdfVector::GdfVector()
{
    column.data = nullptr;
    column.valid = nullptr;
    column.size = 0;
    column.dtype = GDF_invalid;
    column.null_count = 0;
}

GdfVector::GdfVector(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    create_gdf_column(type, num_values, input_data, width_per_value);
}

GdfVector::GdfVector(const GdfVector& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;

    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));
}

GdfVector::GdfVector(GdfVector& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;

    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));
}

void GdfVector::operator=(const GdfVector& col)
{
    column.data = col.column.data;
    column.valid = col.column.valid;
    column.size = col.column.size;
    column.dtype = col.column.dtype;
    column.null_count = col.column.null_count;

    GDFRefCounter::getInstance()->increment(const_cast<gdf_column*>(&col.column));
}

gdf_column* GdfVector::get_gdf_column()
{
    return &column;
}

void GdfVector::create_gdf_column(gdf_dtype type, size_t num_values, void * input_data, size_t width_per_value)
{
    char * data;
    gdf_valid_type * valid_device;

    //@todo: ask someone about that!
    size_t allocation_size_valid = ((((num_values + 7 ) / 8) + 63 ) / 64) * 64; //so allocations are supposed to be 64byte aligned

    cudaMalloc((void **) &valid_device, allocation_size_valid);	

    cudaMemset(valid_device, (gdf_valid_type)255, allocation_size_valid); //assume all relevant bits are set to on

    cudaMalloc((void **) &data, width_per_value * num_values);

    if(input_data != nullptr){
        cudaMemcpy(data, input_data, num_values * width_per_value, cudaMemcpyHostToDevice);
    }

    gdf_column_view(&this->column, (void *) data, valid_device, num_values, type);

    GDFRefCounter::getInstance()->register_column(&this->column);
}

void GdfVector::realloc_gdf_column(gdf_dtype type, size_t size, size_t width){
    GDFRefCounter::getInstance()->decrement(&this->column); //decremeting reference, deallocating space

	create_gdf_column(type, size, nullptr, width);
}

gdf_error GdfVector::gdf_column_view(gdf_column *column, void *data, gdf_valid_type *valid, gdf_size_type size, gdf_dtype dtype)
{
    column->data = data;
    column->valid = valid;
    column->size = size;
    column->dtype = dtype;
    column->null_count = 0;
    return GDF_SUCCESS;
}

GdfVector::~GdfVector()
{
    GDFRefCounter::getInstance()->decrement(&this->column);
}

void* GdfVector::data(){
    return column.data;
}

gdf_valid_type* GdfVector::valid(){
    return column.valid;
}
gdf_size_type GdfVector::size(){
    return column.size;
}

gdf_dtype GdfVector::dtype(){
    return column.dtype;
}

gdf_size_type GdfVector::null_count(){
    return column.null_count;
}

gdf_dtype_extra_info GdfVector::dtype_info(){
    return column.dtype_info;
}

void GdfVector::set_dtype(gdf_dtype dtype){
    column.dtype=dtype;
}

}//container
}//gdf
