
// Define the structure
struct SCand {
    int mt;
    int mdt;
    int mwidth;
    float msnr;
};
using Cand = SCand;

struct CompareCandMember {
    const Cand* d_vctCandHeap;
    int member_offset;

    CompareCandMember(const Cand* _d_vctCandHeap, int _member_offset)
        : d_vctCandHeap(_d_vctCandHeap), member_offset(_member_offset) {}

    __host__ __device__
        bool operator()(const int& idx1, const int& idx2) const {
        const char* base1 = reinterpret_cast<const char*>(&d_vctCandHeap[idx1]);
        const char* base2 = reinterpret_cast<const char*>(&d_vctCandHeap[idx2]);
        int value1 = *reinterpret_cast<const int*>(base1 + member_offset);
        int value2 = *reinterpret_cast<const int*>(base2 + member_offset);

        // Print debug information within the functor
      //  printf("Comparing idx1: %d, value1: %d, idx2: %d, value2: %d\n", idx1, value1, idx2, value2);

        return value1 < value2;
    }
};


struct GetCandMember {
    const Cand* d_vctCandHeap;
    int member_offset;

    GetCandMember(const Cand* _d_vctCandHeap, int _member_offset)
        : d_vctCandHeap(_d_vctCandHeap), member_offset(_member_offset) {}

    __host__ __device__
        int operator()(const int& index) const {
        const char* base = reinterpret_cast<const char*>(&d_vctCandHeap[index]);
        return *reinterpret_cast<const int*>(base + member_offset);
    }
};


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
    , int member_offset
    , thrust::device_vector<int>& d_vctCumSum0
);

void sort_subarrays(const thrust::device_vector<Cand>& d_vctCandHeap,
    thrust::device_vector<int>& h_vctGroupBeginIndices,
    thrust::device_vector<int>& d_indices, int member_offset);

//void update_grouping(const thrust::device_vector<int> d_vctValues
//    , const thrust::device_vector<int> d_vctIndexes
//    , const int& d_bin_metrics, std::vector<int*>& vctGroupPtrs);

__global__
void compute_grouping_plan_kernel(const int* d_arrValues
    , const int quantCandidates
    , int** h_ppGroupPtrs
    , const int quantGroups
    , const int* d_arrIndexes
    , const int& d_bin_metrics
    , int* d_arrGroupingPlan);

__global__
void  carry_out_the_plan(const int* d_arrValues
    , const int quantCandidates
    , int** h_ppGroupPtrs
    , const int quantGroups
    , const int* d_arrIndexes
    , const int& d_bin_metrics
    , int* d_arrGroupingPlan
    , int** h_ppGroupPtrsUpdated);

void fnc_grouping(const thrust::device_vector<Cand> &d_vctCandHeap
    , const int member_offset
    , const int& d_bin_metrics
    , thrust::device_vector<int>& d_vctIndices
    , thrust::host_vector<int>& h_vctGroupBeginIndices);

__global__
void calc_plan_and_values_for_regrouping_kernel(const Cand* d_arrCand
    , const  int* d_arrIndeces
    , const int QUantCand
    , const  int* d_arrGroupBeginIndecies
    , const int member_offset
    , const int& d_bin_metrics
    , int* d_arrValues
    , int* d_arrRegroupingPlan);



__global__
void regrouping_kernel(int* d_vctValues
    , const  int LEn_vctValues
    , const int* d_vctGroupBeginIndices
    , const int* d_vctRegroupingPlan
    , const  int LEn_vctGroupBeginIndices
    , const  int& d_bin_metrics
    , int* d_vctGroupBeginIndicesUpdated);
