#pragma once
//-------------------------------------------------------------------------

void fncFdmt_cpuF_v0(float* piarrImgInp, const int IImgrows, const int IImgcols
	, const float VAlFmin, const  float VAlFmax, const int IMaxDT, float* piarrImgOut);


void fncFdmtIteration_cpuF(float* piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* arr_val0
	, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_MI, int* iarr_dT_ML, int* iarr_dT_RI
	, float* piarrOut, int& iOutPutDim0, int& iOutPutDim1);


void shift_and_sum_cpuF_v1(float* piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* iarr_deltaTLocal, int* iarr_dT_MI
	, int* iarr_dT_ML, int* iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, float* piarrOut);

void shift_and_sum_cpuF(float* piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* iarr_deltaTLocal, int* iarr_dT_MI
	, int* iarr_dT_ML, int* iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, float* piarrOut);


void create_2d_arrays_cpu(const int IDim0, const int IDim1
	, float* arr_val0, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_middle_index, int* iarr_dT_middle_larger
	, int* iarr_dT_rest_index);

void fnc_init_cpuF(float* piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, float* piarrOut);







