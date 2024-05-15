#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <fdmt_gpu.cuh>

#include "npy.hpp" //! delete_

extern cudaError_t cudaStatus0 = cudaSuccess;

FDMTGPU::FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps,
                 float tsamp, size_t dt_max, size_t dt_step, size_t dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min)
{
    // Allocate memory for the state buffers
    const auto& plan      = get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    m_state_in_d.resize(state_size, 0.0F);
    m_state_out_d.resize(state_size, 0.0F);
    transfer_plan_to_device(plan, m_fdmt_plan_d);
}

void FDMTGPU::transfer_plan_to_device(const FDMTPlan& plan, FDMTPlanD& plan_d)
{
    // Transfer the plan to the device
    const auto niter = plan.state_shape.size();
    
    // 1. "state_shape" allocation on GPU
   // plan_d.state_shape_d.resize(niter);
    std::vector<SizeType> state_shape_flattened ;
       
    state_shape_flattened.reserve(niter * 5);
    for (const auto& array : plan.state_shape)
    {        
        state_shape_flattened.insert(state_shape_flattened.end(), array.begin(), array.end());
    }
    plan_d.state_shape_d.resize(state_shape_flattened.size());
    thrust::copy(state_shape_flattened.begin(), state_shape_flattened.end(), plan_d.state_shape_d.begin());
    
    // !1

    // 2. "coordinates_d" allocation on GPU
      // 2.1 "lenof_innerVects_coords_cumsum" creation and  allocation on GPU
    std::vector< SizeType>h_lenof_innerVects_coords_cumsum(niter +1);
    h_lenof_innerVects_coords_cumsum[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_lenof_innerVects_coords_cumsum[i] = h_lenof_innerVects_coords_cumsum[i - 1] + plan.coordinates[i - 1].size() *2;
    }
    plan_d.lenof_innerVects_coords_cumsum.resize(h_lenof_innerVects_coords_cumsum.size());
    thrust::copy(h_lenof_innerVects_coords_cumsum.begin(), h_lenof_innerVects_coords_cumsum.end(), plan_d.lenof_innerVects_coords_cumsum.begin());
    // !2.1

    // 2.2
    size_t totalSize = h_lenof_innerVects_coords_cumsum[h_lenof_innerVects_coords_cumsum.size() - 1];
    std::vector< SizeType> h_coordinates_flattened(totalSize);   
    
    for (const auto& innerVec : plan.coordinates) {
        for (const auto& pair : innerVec) {
            h_coordinates_flattened.push_back(pair.first);
            h_coordinates_flattened.push_back(pair.second);
        }
    }
    plan_d.coordinates_d.resize(h_coordinates_flattened.size());
    thrust::copy(h_coordinates_flattened.begin(), h_coordinates_flattened.end(), plan_d.coordinates_d.begin());   
        // !2.2
    //!.2

    // 3 "coordinates_to_copy" allocation on GPU
          // 3.1 "lenof_innerVects_coords_to_copy_cumsum" creation and allocation on GPU
    std::vector< SizeType>h_lenof_innerVects_coords_to_copy_cumsum(niter + 1);
    h_lenof_innerVects_coords_to_copy_cumsum[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_lenof_innerVects_coords_to_copy_cumsum[i] = h_lenof_innerVects_coords_to_copy_cumsum[i - 1] + plan.coordinates_to_copy[i - 1].size() * 2;
    }
    plan_d.lenof_innerVects_coords_to_copy_cumsum.resize(h_lenof_innerVects_coords_to_copy_cumsum.size());
    thrust::copy(h_lenof_innerVects_coords_to_copy_cumsum.begin(), h_lenof_innerVects_coords_to_copy_cumsum.end(), plan_d.lenof_innerVects_coords_to_copy_cumsum.begin());  
    // !3.1

    // 3.2
    totalSize = h_lenof_innerVects_coords_to_copy_cumsum[h_lenof_innerVects_coords_to_copy_cumsum.size() - 1];
    std::vector< SizeType> h_coordinates_to_copy_flattened(totalSize);

    for (const auto& innerVec : plan.coordinates_to_copy) {
        for (const auto& pair : innerVec) {
            h_coordinates_to_copy_flattened.push_back(pair.first);
            h_coordinates_to_copy_flattened.push_back(pair.second);
        }
    }
    plan_d.coordinates_to_copy_d.resize(h_coordinates_to_copy_flattened.size());
    thrust::copy(h_coordinates_to_copy_flattened.begin(), h_coordinates_to_copy_flattened.end(), plan_d.coordinates_to_copy_d.begin()); 
    //!3
    
    // 4. "mappings" allocation on GPU
        // 4.1 "len_mappings_cumsum" creation and allocation on GPU
    std::vector< SizeType>h_len_mappings_cumsum(niter + 1);
    h_len_mappings_cumsum[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_len_mappings_cumsum[i] = h_len_mappings_cumsum[i - 1] + plan.mappings[i - 1].size() * 5;
    }
    plan_d.len_mappings_cumsum.resize(h_len_mappings_cumsum.size());
    thrust::copy(h_len_mappings_cumsum.begin(), h_len_mappings_cumsum.end(), plan_d.len_mappings_cumsum.begin()); 
    // !4.1

    // 4.2
    std::vector<SizeType> mappings_flattened = flatten_mappings(plan.mappings);
    plan_d.mappings_d.resize(mappings_flattened.size());
    thrust::copy(mappings_flattened.begin(), mappings_flattened.end(), plan_d.mappings_d.begin()); 
    //!4.2, !4    

    // 5. "mappings_to_copy" allocation on GPU
        // 5.1 "len_mappings_to_copy_cumsum" creation and allocation on GPU
    std::vector< SizeType>h_len_mappings_to_copy_cumsum(niter + 1);
    h_len_mappings_to_copy_cumsum[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_len_mappings_to_copy_cumsum[i] = h_len_mappings_to_copy_cumsum[i - 1] + plan.mappings_to_copy[i - 1].size() * 5;
    }
    plan_d.len_mappings_to_copy_cumsum.resize(h_len_mappings_to_copy_cumsum.size());
    thrust::copy(h_len_mappings_to_copy_cumsum.begin(), h_len_mappings_to_copy_cumsum.end(), plan_d.len_mappings_to_copy_cumsum.begin());
    // !5.1

    // 5.2
    std::vector<SizeType> mappings_to_copy_flattened = flatten_mappings(plan.mappings_to_copy);
    plan_d.mappings_to_copy_d.resize(mappings_to_copy_flattened.size());
    thrust::copy(mappings_to_copy_flattened.begin(), mappings_to_copy_flattened.end(), plan_d.mappings_to_copy_d.begin());
    //!5.2, !5

    // 6."state_sub_idx_d" allocation on GPU
        // 6.1 "len_state_sub_idx_cumsum" creation and allocation on GPU
    std::vector< SizeType>h_len_state_sub_idx_cumsum(niter + 1);
    h_len_state_sub_idx_cumsum[0] = 0;
    for (int i = 1; i < niter + 1; ++i)
    {
        h_len_state_sub_idx_cumsum[i] = h_len_state_sub_idx_cumsum[i - 1] + plan.state_sub_idx[i - 1].size() ;
    }
    plan_d.len_state_sub_idx_cumsum.resize(h_len_state_sub_idx_cumsum.size());
    thrust::copy(h_len_state_sub_idx_cumsum.begin(), h_len_state_sub_idx_cumsum.end(), plan_d.len_state_sub_idx_cumsum.begin());
    // !6.1

    //6.2
    totalSize = h_len_state_sub_idx_cumsum[h_len_state_sub_idx_cumsum.size() - 1];
    plan_d.state_sub_idx_d.resize(totalSize);

    std::vector<SizeType> flattened;
    for (const auto& innerVec : plan.state_sub_idx) {
        flattened.insert(flattened.end(), innerVec.begin(), innerVec.end());
    }
    thrust::copy(flattened.begin(), flattened.end(), plan_d.state_sub_idx_d.begin());
    
    // !6.2, !6

    // 7. "dt_grid_d" allocation on GPU
        // 7.1 CPU preparations
    std::vector<SizeType> dt_grid_flattened;
    // Vector to store the start indices of each DtGridType
    std::vector<SizeType> pos_gridInnerVects_h;
    // Vector to store the start indices of each vector<DtGridType>
    std::vector<SizeType> pos_gridSubVects_h;

    SizeType currentIndex = 0;

    for (const auto& subVect : plan.dt_grid)
    {
        // Save the starting point of each subvector in dt_grid
        pos_gridSubVects_h.push_back(currentIndex);

        for (const auto& dtGrid : subVect) 
        {
            // Save the starting point of each DtGridType
            pos_gridInnerVects_h.push_back(currentIndex);

            // Append elements of dtGrid to the flattened vector
            dt_grid_flattened.insert(dt_grid_flattened.end(), dtGrid.begin(), dtGrid.end());
            currentIndex += dtGrid.size();
        }
    }

    pos_gridSubVects_h.push_back(pos_gridInnerVects_h.size());
    pos_gridInnerVects_h.push_back(dt_grid_flattened.size());

    // delete_
    int s = 0;
    int s1 = 0;
    for (int i = 0; i < 512; ++i)
    {
        s += pos_gridInnerVects_h[dt_grid_flattened[0] + i + 1] - pos_gridInnerVects_h[dt_grid_flattened[0] + i];
        s1 += plan.dt_grid[0][i].size();
        std::cout << "i_dt = " << pos_gridInnerVects_h[dt_grid_flattened[0] + i + 1] - pos_gridInnerVects_h[dt_grid_flattened[0] + i] << std::endl;
    }
    std::cout << "s = " << s << "   s1 = " << s1 << std::endl;

    std::cout << "pos_gridInnerVects_h[dt_grid_flattened[0] +511] = " << pos_gridInnerVects_h[dt_grid_flattened[0] + 511] << std::endl;
    std::cout << "pos_gridInnerVects_h[dt_grid_flattened[0] +512] = " << pos_gridInnerVects_h[dt_grid_flattened[0] + 512] << std::endl;
    
        // 7.2 GPU allocation
    plan_d.dt_grid_d.resize(dt_grid_flattened.size());
    plan_d.pos_gridSubVects.resize(pos_gridSubVects_h.size());
    plan_d.pos_gridInnerVects.resize(pos_gridInnerVects_h.size());

    thrust::copy(dt_grid_flattened.begin(), dt_grid_flattened.end(), plan_d.dt_grid_d.begin());
    thrust::copy(pos_gridSubVects_h.begin(), pos_gridSubVects_h.end(), plan_d.pos_gridSubVects.begin());
    thrust::copy(pos_gridInnerVects_h.begin(), pos_gridInnerVects_h.end(), plan_d.pos_gridInnerVects.begin());
    // !7
}
//------------------------------------------------------------
//  Function to flatten the vector of vectors of FDMTCoordMapping
std::vector<SizeType> flatten_mappings(const std::vector<std::vector<FDMTCoordMapping>>& mappings)
{
    std::vector<SizeType> flattened;
    
    size_t totalSize = 0;
    for (const auto& vec : mappings) {
        totalSize += vec.size() * 5;  // Each FDMTCoordMapping has 5 SizeType elements
    }
    flattened.reserve(totalSize);
    
    for (const auto& innerVec : mappings) {
        for (const auto& mapping : innerVec) {
            flattened.push_back(mapping.head.first);
            flattened.push_back(mapping.head.second);
            flattened.push_back(mapping.tail.first);
            flattened.push_back(mapping.tail.second);
            flattened.push_back(mapping.offset);
        }
    }

    return flattened;
}

//-----------------------------------------------------
void  FDMTGPU::execute(const float* waterfall, size_t waterfall_size, float* dmt,  size_t dmt_size)
{
    check_inputs(waterfall_size, dmt_size);
   
    float* state_in_ptr = thrust::raw_pointer_cast(m_state_in_d.data());   
    float* state_out_ptr = thrust::raw_pointer_cast(m_state_out_d.data());

    initialise(waterfall, state_in_ptr);

    const auto& plan = get_plan();
    
    int lenarr4 = plan.state_shape[0][3] * plan.state_shape[0][4];
    std::vector<float> data4(lenarr4, 0);
    cudaMemcpy(data4.data(), state_in_ptr, lenarr4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::array<long unsigned, 1> leshape2{ lenarr4 };
    npy::SaveArrayAsNumpy("state_gpu.npy", false, leshape2.size(), leshape2.data(), data4);
   /* const auto niters = get_niters();
    for (size_t i_iter = 1; i_iter < niters + 1; ++i_iter) {
        execute_iter(state_in_ptr, state_out_ptr, i_iter);
        if (i_iter < niters) {
            std::swap(state_in_ptr, state_out_ptr);
        }
    }
    std::copy_n(state_out_ptr, dmt_size, dmt);*/
}
//--------------------------------------------------
void FDMTGPU::initialise(const float* waterfall, float* state)
{
    const auto& plan = get_plan();
    const auto& dt_grid_init = plan.dt_grid[0];
    const auto& state_sub_idx_init = plan.state_sub_idx[0];
    const auto& nsamps = plan.state_shape[0][4];
    const long long nchan = dt_grid_init.size();

    SizeType* p_state_sub_idx = thrust::raw_pointer_cast(m_fdmt_plan_d.state_sub_idx_d.data());
    SizeType* p_len_state_sub_idx_cumsum = thrust::raw_pointer_cast(m_fdmt_plan_d.len_state_sub_idx_cumsum.data());
    SizeType* p_dt_grid = thrust::raw_pointer_cast(m_fdmt_plan_d.dt_grid_d.data());
    SizeType* p_pos_gridSubVects = thrust::raw_pointer_cast(m_fdmt_plan_d.state_sub_idx_d.data());
    SizeType*  p_pos_gridInnerVects = thrust::raw_pointer_cast(m_fdmt_plan_d.pos_gridSubVects.data());
    //m_fdmt_plan_d
    const dim3 blockSize = dim3(1024, 1);
    const dim3 gridSize = dim3((nsamps + blockSize.x - 1) / blockSize.x, nchan );
    kernel_init_fdmt << < gridSize, blockSize >> > (waterfall,  p_state_sub_idx, p_len_state_sub_idx_cumsum, p_dt_grid, p_pos_gridSubVects
       ,  p_pos_gridInnerVects, state, nsamps);
    cudaDeviceSynchronize();
    cudaStatus0 = cudaGetLastError();
    if (cudaStatus0 != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus0));
        return;
    }
    /*int lenarr4 = msamp * m_nchan * m_len_sft;
            std::vector<float> data4(lenarr4, 0);
            cudaMemcpy(data4.data(), (float*)m_pdbuff_rolled, lenarr4 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            std::array<long unsigned, 1> leshape2{ lenarr4 };
            npy::SaveArrayAsNumpy("parr_wfall_py.npy", false, leshape2.size(), leshape2.data(), data4);
            int ii = 0;*/
    
}
//--------------------------------------------------------------

__global__
void  kernel_init_fdmt(const float* waterfall, SizeType* p_state_sub_idx, SizeType* p_len_state_sub_idx_cumsum
    , SizeType* p_dt_grid, SizeType* p_pos_gridSubVects    , SizeType* p_pos_gridInnerVects, float* state, const int nsamps)
{    
    SizeType isamp = blockIdx.x * blockDim.x + threadIdx.x;
    if (isamp >= nsamps)
    {
    return;
    }
    SizeType  i_sub = blockIdx.y;  
    
   
    //// Initialise state for [:, dt_init_min, dt_init_min:]
  
    SizeType  dt_grid_sub_min = p_dt_grid[p_pos_gridInnerVects[p_pos_gridSubVects[0] + i_sub]];   
   
    SizeType  state_sub_idx = p_state_sub_idx[p_len_state_sub_idx_cumsum[0] + i_sub];
   
    if (isamp < dt_grid_sub_min)
    {
        return;
    }           
    float sum = 0.0F;
    for (size_t i = isamp - dt_grid_sub_min; i <= isamp; ++i)
    {
       sum += waterfall[i_sub * nsamps + i];
    }
    state[state_sub_idx + isamp] =  sum / static_cast<float>(dt_grid_sub_min + 1);
    ////---
    SizeType  dt_grid_sub_size =  p_pos_gridInnerVects[p_pos_gridSubVects[0] + i_sub + 1] - p_pos_gridInnerVects[p_pos_gridSubVects[0] + i_sub];
    if (dt_grid_sub_size > 10)
    {
        printf("!! i_sub =  % i    dt_grid_sub_size = %i\n", i_sub , dt_grid_sub_size);
    }
    
        
        //for (size_t i_dt = 1; i_dt < dt_grid_sub_size; ++i_dt)
        //{
        //    
        //    
        //    const auto dt_cur = p_dt_grid[p_pos_gridInnerVects[p_pos_gridSubVects[0] + i_sub] + i_dt];// dt_grid_sub[i_dt];
        //    const auto dt_prev =  p_dt_grid[p_pos_gridInnerVects[p_pos_gridSubVects[0] + i_sub] + i_dt - 1];
        //     float sum = 0.0F;
        //     if (isamp >= dt_cur)
        //     {
        //         for (size_t i = isamp - dt_cur; i < isamp - dt_prev; ++i)
        //         {
        //             sum += waterfall[i_sub * nsamps + i];
        //         }
        //         state[state_sub_idx + i_dt * nsamps + isamp] = (state[state_sub_idx + (i_dt - 1) * nsamps + isamp] *
        //         (static_cast<float>(dt_prev) + 1.0F) +  sum) /(static_cast<float>(dt_cur) + 1.0F);
        //     }
        //     else
        //     {
        //         state[state_sub_idx + i_dt * nsamps + isamp] = 0.0F;
        //     }
        //}
   
}
//--------------------------------------------------------------------------------------
__global__
void kernel_init_fdmt0(float* d_parrImg, const int& IImgrows, const int* IImgcols
    , const int& IDeltaTP1, float* d_parrOut, const bool b_ones)
{
    int i_F = blockIdx.y;
    int numOutElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (numOutElemInRow >= (*IImgcols))
    {
        return;
    }
    int numOutElemPos = i_F * IDeltaTP1 * (*IImgcols) + numOutElemInRow;
    int numInpElemPos = i_F * (*IImgcols) + numOutElemInRow;
    float  itemp = (b_ones) ? 1.0f : (float)d_parrImg[numInpElemPos];
    d_parrOut[numOutElemPos] = (float)itemp;
    //printf("init");
    // old variant
    for (int i_dT = 1; i_dT < IDeltaTP1; ++i_dT)
    {
        numOutElemPos += (*IImgcols);
        if (i_dT <= numOutElemInRow)
        {
            float  val = (b_ones) ? 1.0 : ((float)d_parrImg[i_F * (*IImgcols) + numOutElemInRow - i_dT]);
            itemp = fdividef(fmaf(itemp, (float)i_dT, val), (i_dT + 1));
            d_parrOut[numOutElemPos] = (float)itemp;
        }
        else
        {
            d_parrOut[numOutElemPos] = 0;
        }
    }

}
