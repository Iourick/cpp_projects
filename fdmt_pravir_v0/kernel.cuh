#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//
//void fncFdmt_cu_v0(int* piarrImg, const int iImgrows
//	, const int iImgcols, const float f_min
//	, const  float f_max, const int imaxDT, int* piarrOut);
//
//void fnc_init(int* d_piarrImg, const int IImgrows, const int IImgcols
//	, const int IDeltaT, int* d_piarrOut);
//
//void fncFdmtIteration(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
//	, const int IDim2, const int IMaxDT, const float VAlFmin
//	, const float VAlFmax, const int ITerNum, float* d_arr_val0
//	, float* d_arr_val1, int* d_iarr_deltaTLocal
//	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
//	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1);
//
//__global__
//void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
//	, const float VAlc2, const float VAlf_min, const float VAlcorrection
//	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal);
//
//__global__
//void kernel_2d_arrays(const int IDim0, const int IDim1
//	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
//	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
//	, int* d_iarr_dT_rest_index);
//
//__global__
//void kernel_shift_and_sum(int* d_piarrInp, const int IDim0, const int IDim1
//	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
//	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
//	, int* d_piarrOut);
//
//__global__ void sumArrays(int* d_result, const int* d_arr1, const int* d_arr2, int n);
//
//__global__ void sumArrays_(int* d_result, const int* d_arr1, int n);
//
//void fncCalcDimensionsOfOutputArrays(std::vector<int>* pivctOutDim0, std::vector<int>* pivctOutDim1
//	, std::vector<int>* pivctOutDim2, const int IDim0, const int IDim1
//	, const int IDim2, const int IMaxDT, const float VAlFmin
//	, const float VAlFmax);
//void shift_and_sum(int* d_piarrInp, const int IDim0, const int IDim1
//	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
//	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
//	, int* d_piarrOut);
//
//
//void fnc_init_fdmt(int* d_piarrImg, const int IImgrows, const int IImgcols
//	, const int IDeltaTplus1, int* d_piarrOut);
//
//__global__
//void kernel_seed(int* d_piarrImg, const int IImgrows, const int IImgcols
//	, const int IDeltaT, int* d_piarrOut);
//
//__global__
//void init_iter(int* d_piarrImg, const int IImgrows, const int IImgcols, const int IDeltaT
//	, const int i_dT, int* d_piarrOut);
