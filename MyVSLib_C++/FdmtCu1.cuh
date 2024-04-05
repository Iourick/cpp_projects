#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void fncFdmt_cu_v1(int* piarrImage // input image
	, int* d_piarrImage       // on-device auxiliary memory buffer
	, const int IImgrows, const int IImgcols // dimensions of input image 	
	, int* d_piarrState0		// on-device auxiliary memory buffer
	, int* d_piarrState1		// on-device auxiliary memory buffer
	, const int IDeltaT
	, const int I_F
	, const float VAl_dF
	, float* d_arr_val0 		// on-device auxiliary memory buffer
	, float* d_arr_val1			// on-device auxiliary memory buffer
	, int* d_arr_deltaTLocal	// on-device auxiliary memory buffer
	, int* d_arr_dT_MI			// on-device auxiliary memory buffer
	, int* d_arr_dT_ML			// on-device auxiliary memory buffer
	, int* d_arr_dT_RI			// on-device auxiliary memory buffer
	, const  float VAlFmin, const  float VAlFmax, const int IMaxDT
	, int* u_piarrImOut			// OUTPUT image, dim = IDeltaT x IImgcols
);

__global__
void kernel_init_iter_v1(int* d_piarrImgRow, const int IImgrows, const int IImgcols
	, const int i_dT, int* d_pMtrxPrev, int* d_pMtrxCur);

void fncFdmtIteration_v1(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1);

__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal);




void fnc_init_fdmt_v1(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaTplus1, int* d_piarrOut);

__global__
void kernel_create_aux_2d_arrays_v1(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_iarr_dT_rest_index);

__global__
void kernel3D_shift_and_sum_v1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel1D_shift_and_sum_v1(const int quantBlocksPerRow, int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel1D_shift_and_sum_v11(const int quantBlocksPerRow, int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1/*, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI*/, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel1D_shift_and_sum_v11(const int quantBlocksPerRow, int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1/*, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI*/, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel3D_shift_and_sum_v11(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel3D_shift_and_sum_v12(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel_shift_and_sum_(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

__global__
void kernel2D_shift_and_sum_v1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);


__global__
void kernel_init_yk0(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_pMtrxCur);


__global__
void kernel_init_yk1(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrState0);





