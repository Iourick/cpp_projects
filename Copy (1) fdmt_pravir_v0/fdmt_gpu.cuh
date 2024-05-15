#pragma once

#include <thrust/device_vector.h>

#include <fdmt_base.hpp>

using StShapeTypeD   = int4;
using FDMTCoordTypeD = int2;

struct FDMTCoordMappingD {
    FDMTCoordTypeD head;
    FDMTCoordTypeD tail;
    SizeType offset;
};

//struct FDMTPlanD {
//    thrust::device_vector<SizeType> nsubs_d;
//    thrust::device_vector<SizeType> ncoords_d;
//    thrust::device_vector<SizeType> ncoords_to_copy_d;
//    thrust::device_vector<SizeType> nsubs_cumul_d;
//    thrust::device_vector<SizeType> ncoords_cumul_d;
//    thrust::device_vector<SizeType> ncoords_to_copy_cumul_d;
//    // i = i_iter
//    thrust::device_vector<StShapeTypeD> state_shape_d;
//    // i = i_iter * ncoords_cumul_iter + i_coord
//    thrust::device_vector<FDMTCoordTypeD> coordinates_d;
//    thrust::device_vector<FDMTCoordMappingD> mappings_d;
//    // i = i_iter * ncoords_to_copy_cumul_iter + i_coord_to_copy
//    thrust::device_vector<FDMTCoordTypeD> coordinates_to_copy_d;
//    thrust::device_vector<FDMTCoordMappingD> mappings_to_copy_d;
//    // i = i_iter * nsubs_cumul_iter + isub
//    thrust::device_vector<SizeType> state_sub_idx_d;
//};
struct FDMTPlanD
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
    thrust::device_vector<SizeType> lenof_innerVects_coords_cumsum;

    // Is an analogues as previous
    thrust::device_vector<SizeType> coordinates_to_copy_d;
    thrust::device_vector<SizeType> lenof_innerVects_coords_to_copy_cumsum;


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
    thrust::device_vector<SizeType> len_state_sub_idx_cumsum;


    // It is analogue of dt_grid
    // Has size: dt_grid_d.size() = m_niters +1
    thrust::device_vector<SizeType>dt_grid_d;
    thrust::device_vector<SizeType> pos_gridSubVects;
    thrust::device_vector<SizeType> pos_gridInnerVects;
};
class FDMTGPU : public FDMT {
public:
    FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
            size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
    void execute(const float* waterfall, size_t waterfall_size, float* dmt,
                 size_t dmt_size) override;
    void initialise(const float* waterfall, float* state) override;

private:
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;

    FDMTPlanD m_fdmt_plan_d;

    static void transfer_plan_to_device(const FDMTPlan& plan,
                                        FDMTPlanD& plan_d);
};

std::vector<SizeType> flatten_mappings(const std::vector<std::vector<FDMTCoordMapping>>& mappings);

__global__
void kernel_init_fdmt0(float * d_parrImg, const int& IImgrows, const int* IImgcols
    , const int& IDeltaTP1, float * d_parrOut, const bool b_ones);

__global__
void  kernel_init_fdmt(const float* waterfall, SizeType* p_state_sub_idx, SizeType* p_len_state_sub_idx_cumsum
    , SizeType* p_dt_grid, SizeType* p_pos_gridSubVects, SizeType* p_pos_gridInnerVects, float* state, const int nsamps);
