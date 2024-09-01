#ifndef CANDIDATE_CUH
#define CANDIDATE_CUH
// Define the structure
struct SCand {
    int mt;
    int mdt;
    int mwidth;
    float msnr;
};
using Cand = SCand;

// Functor to digitize the values
struct DigitizeFunctor {
    float max_value;

    DigitizeFunctor(float max_value) : max_value(max_value) {}

    __host__ __device__
        int operator()(const float& x) const {
        return static_cast<int>(x * (1 << 11) / max_value);
    }
};

//----------------------------------------------------------

struct CompareCand {
    const Cand* data;

    CompareCand(const Cand* data) : data(data) {}

    __device__ bool operator()(int a, int b) const {
        return data[a].msnr < data[b].msnr;
    }
};

struct CompareCandMember
{
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
        return value1 < value2;
    }
};

//--------------------------------
inline bool compareByMt(const Cand& a, const Cand& b)
{
    return a.mt < b.mt;
}

#endif // CANDIDATE_CUH