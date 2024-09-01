#ifndef STATISTICAL_PREPROCESSING_CUH
#define STATISTICAL_PREPROCESSING_CUH
#include "Constants.h"
namespace statistical_preprocessing
{
    void clean_fdmtInput_and_create_normInput(float* d_arr, const int NRows, const int NCols
        , bool* d_pbarrNorm, float* d_buff);

	void calc_fdmt_inputs(float* d_parr_inp, const int NRows, const int NCols
		, float* d_buff, float* d_parrNormInput);

	inline int calcThreadsForMean_and_Disp(unsigned const int nCols)
	{
		int k = std::log(nCols) / std::log(2.0);
		k = ((1 << k) > nCols) ? k + 1 : k;
		return 1 << std::min(k, 10);
	};
};
__global__
void calculateMeanAndSTD_for_oneDimArray_kernel(float* d_arr, const unsigned int len, float* pmean
	, float* pstd); //+

__global__
void kernel_OneSM_Mean_and_Std(float* d_arrMeans, float* d_arrDisps, int len
	, float* pmean0, float* pstd); //+

__global__ void calcRowMeanAndDisp(float* d_arrIm, int nRows, int nCols
	, float* arrSumMean, float* arrDisps); //+

__global__
void normalize_clean_and_fill_NormInput(float* d_arr, const int NCols
	, float* pmean, float* pstd, float* d_arrRowDisp, float* pmeanDisp, float* pstdDisp
	, float* d_parrNormInput);//+

void cleanInpFDMT_v0(float* d_arr, const int NRows, const int NCols, float* d_buff);

__global__ 
void calcRowDisps_kernel(float* d_arrIm, const int nRows, const int nCols, float* arrDisps);

__global__
void
calculateMeanAndSTD_kernel(float* d_arr, const unsigned int len, float* pmean, float* pstd);

__global__
void clean_out_the_trash_kernel_v2(float* d_arr, const int NRows, const int NCols, float* d_buff, const float mean, const float std);

__global__ 
void kernel_calcDispersion_for_Short_Onedimentional_Array(float* d_poutDisp, const float* d_arrInp, const float* d_mean, const int len);

__global__ 
void kernel_normalize_array(fdmt_type_* pAuxBuff, const unsigned int len
	, float* pmean, float* pdev, float* parrInp);





__global__
void normalize_and_clean(fdmt_type_* parrOut, float* d_arr, const int NRows, const int NCols
	, float* pmean, float* pstd, float* d_arrRowDisp, float* pmeanDisp, float* pstdDisp);




#endif // STATISTICAL_PREPROCESSING_CUH