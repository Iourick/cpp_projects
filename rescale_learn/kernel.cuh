
#include <cstddef>
#include <vector>
void fill_inpVect(std::vector<uint8_t >& vct_inpbuf, const int  nt, const int nf, const bool  invert_freq);

float calculateMean(float* arr, int n);

float calculateVariance(float* arr, int n, float mean);