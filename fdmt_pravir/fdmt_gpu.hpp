#pragma once

#include <thrust/device_vector.h>

#include <fdmt_base.hpp>

struct FDMTPlanD {
    thrust::device_vector<SizeType> state_shape_d;
    thrust::device_vector<SizeType> state_idx_d;
    thrust::device_vector<SizeType> dt_grid_d;
    thrust::device_vector<SizeType> dt_plan_d;
};
struct FDMTPlanD_
{
    // lets implement variable : const int NUmIter = m_niters +1;
    // it is vector consisting of inner vectors. each inner vector contains 5 elements type of SizeType
    // so, lets declare : size_t length = state_shape_d.size();
    // keeping in mind that structure StShapeType consists of 5 elements, we can reach element j of 
    // prototype state_shape  by j*5 indexing
    thrust::device_vector<SizeType> state_shape_d;

    // is analogue of std::vector<std::vector<FDMTCoordType>> coordinates;
    // "coordinates_d" is flattened vector "coordinates" 
    // in vector "len_inner_vects_coordinates_cumsum" we store cummulative sums of elements  inner vectors of "coordinates" 
    // So, 
    // len_inner_vects_coordinates_cumsum[0] = 0
    // len_inner_vects_coordinates_cumsum[1] = len_inner_vects_coordinates_cumsum[0] + coordinates[0].size() *2
    // ...
    // len_inner_vects_coordinates_cumsum[n] = len_inner_vects_coordinates_cumsum[n-1] + coordinates[n-1].size() *2
    // ...
    // Remember that always:  len_inner_vects_coordinates_cumsum.size() = NUmIter +1
    thrust::device_vector<SizeType> coordinates_d;
    thrust::device_vector<SizeType> len_inner_vects_coordinates_cumsum;

    // Is an analogues as previous
    thrust::device_vector<SizeType> coordinates_to_copy_d;
    thrust::device_vector<SizeType> len_inner_vects_coordinates_to_copy_cumsum;


    // It is analogue of:   std::vector<std::vector<FDMTCoordMapping>> mappings;
    // each FDMTCoordMapping consists of 5 elements
    // "mappings_d" is flattened vector "mappings" 
    // in vector "len_mappings_cumsum" we store cummulative sums of elements  inner vectors of "mappings" 
    // So, 
    // len_mappings_cumsum[0] = 0
    // len_mappings_cumsum[1] = len_mappings_cumsum[0] + mappings[0].size() *5
    // ...
    // len_mappings_cumsum[n] = len_mappings_cumsum[n-1] + mappings[n-1].size() *5
    // ...
    // Remember that always:  len_mappings_cumsum.size() = NUmIter +1
    thrust::device_vector<SizeType> mappings_d;
    thrust::device_vector<SizeType> len_mappings_cumsum;


    // Is an analogues as previous
    thrust::device_vector<SizeType> mappings_to_copy_d;
    thrust::device_vector<SizeType> len_mappings_to_copy_cumsum;


    // It is analogue of state_sub_idx
    // Has size: state_sub_idx_d.size() = m_niters +1
    thrust::device_vector<SizeType>state_sub_idx_d;


    // It is analogue of dt_grid
    // Has size: dt_grid_d.size() = m_niters +1
    thrust::device_vector<SizeType>dt_grid_d;    
};


class FDMTGPU : public FDMT {
public:
    FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
            size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
   /* void execute(const float* waterfall, size_t waterfall_size, float* dmt,
                 size_t dmt_size) override;
    void initialise(const float* waterfall, float* state) override;*/

private:
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;

    FDMTPlanD m_fdmt_plan_d;

    FDMTPlanD transfer_plan_to_device();
    
};