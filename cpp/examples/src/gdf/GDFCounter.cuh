/*
 * GDFCounter.h
 *
 *  Created on: Sep 12, 2018
 *      Author: rqc
 */

#ifndef GDFCOUNTER_H_
#define GDFCOUNTER_H_

#include <map>
#include <mutex>

#include "libgdf.h"

namespace libgdf{

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

}

#endif /* GDFCOUNTER_H_ */