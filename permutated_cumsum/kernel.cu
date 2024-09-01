#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <iostream>
#include <vector>

typedef struct SCand {
    int mt;
    int mdt;
    int mwidth;
} Cand;
//-----------------------------------------------------
void checkCudaErrors(const char* label)
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << label << ": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}
struct FetchMemberValues
{
    const Cand* d_vctCandHeap;
    const int* d_indices;
    int member_offset;
    int heap_size;

    FetchMemberValues(const Cand* _d_vctCandHeap, const int* _d_indices, int _member_offset, int _heap_size)
        : d_vctCandHeap(_d_vctCandHeap), d_indices(_d_indices), member_offset(_member_offset), heap_size(_heap_size) {}

    __host__ __device__
        int operator()(const int& idx) const
    {
        int index = d_indices[idx];
        if (index < 0 || index >= heap_size)
        {
           // printf("Index out of bounds: %d, heap_size: %d\n", index, heap_size);
            return 0; // or some other error handling
        }
        const char* base = reinterpret_cast<const char*>(&d_vctCandHeap[index]);
        int value = *reinterpret_cast<const int*>(base + member_offset);
       // printf("Index: %d, Value: %d\n", index, value); // Debug print
        return value;
    }
};

void cumsum_of_permutated_indexes(const thrust::device_vector<Cand> d_vctCandHeap
    , const thrust::device_vector<int> d_indices
    ,int member_offset
    ,thrust::device_vector<int>& d_vctCumSum0
)
{
    thrust::device_vector<int> temp(d_indices.size());
    int heap_size = static_cast<int>(d_vctCandHeap.size());
    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(d_indices.size()),
        temp.begin(),
        FetchMemberValues(thrust::raw_pointer_cast(d_vctCandHeap.data()),
            thrust::raw_pointer_cast(d_indices.data()),
            member_offset,
            heap_size)
    );
    d_vctCumSum0[0] = 0;

    thrust::inclusive_scan(
        temp.begin(),
        temp.end(),
        d_vctCumSum0.begin() + 1
    );
}



int main() {
    int quant_candidates = 5;

    std::vector<Cand> h_vctCandHeap = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11, 12},
        {13, 14, 15}
    };

    std::vector<int> h_indices = { 4, 3, 2, 1, 0 };

    thrust::device_vector<Cand> d_vctCandHeap = h_vctCandHeap;
    thrust::device_vector<int> d_indices = h_indices;

    std::cout << "Indices: ";
    for (int i = 0; i < d_indices.size(); i++) {
        std::cout << d_indices[i] << " ";
    }
    std::cout << std::endl;

    int heap_size = static_cast<int>(d_vctCandHeap.size());
    int indices_size = static_cast<int>(d_indices.size());
    std::cout << "Heap size: " << heap_size << std::endl;

    int member_offset = offsetof(Cand, mdt);

    thrust::device_vector<int> d_vctCumSum0(d_indices.size() + 1);
    cumsum_of_permutated_indexes( d_vctCandHeap
        ,  d_indices
        , member_offset
        , d_vctCumSum0
    );
    /*thrust::device_vector<int> temp(d_indices.size());

    try {
        thrust::transform(
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(d_indices.size()),
            temp.begin(),
            FetchMemberValues(thrust::raw_pointer_cast(d_vctCandHeap.data()),
                thrust::raw_pointer_cast(d_indices.data()),
                member_offset,
                heap_size)
        );
        checkCudaErrors("thrust::transform");
    }
    catch (thrust::system_error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    d_vctCumSum0[0] = 0;

    try {
        thrust::inclusive_scan(
            temp.begin(),
            temp.end(),
            d_vctCumSum0.begin() + 1
        );
        checkCudaErrors("thrust::inclusive_scan");
    }
    catch (thrust::system_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    */

    thrust::host_vector<int> h_vctCumSum0 = d_vctCumSum0;

    for (int i = 0; i < h_vctCumSum0.size(); i++) {
        std::cout << "d_vctCumSum0[" << i << "] = " << h_vctCumSum0[i] << std::endl;
    }

    return 0;
}
