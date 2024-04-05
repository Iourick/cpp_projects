#pragma once
#include "Constants.h"
//-------------------------------------------------------------------------
//#define fdmt_type_ float
void fncFdmt_cpuT_v0(fdmt_type_ * piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, fdmt_type_ * piarrOut);


void fnc_init_cpuT(fdmt_type_* piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, fdmt_type_ * piarrOut);


void fncFdmtIteration_cpuT(fdmt_type_ * piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* arr_val0
	, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_MI, int* iarr_dT_ML, int* iarr_dT_RI
	, fdmt_type_ * piarrOut, int& iOutPutDim0, int& iOutPutDim1);


void create_2d_arrays_cpu(const int IDim0, const int IDim1
	, float* arr_val0, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_middle_index, int* iarr_dT_middle_larger
	, int* iarr_dT_rest_index);


void shift_and_sum_cpuT_v1(fdmt_type_ * piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* iarr_deltaTLocal, int* iarr_dT_MI
	, int* iarr_dT_ML, int* iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, fdmt_type_ * piarrOut);


void shift_and_sum_cpuT(fdmt_type_ * piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* iarr_deltaTLocal, int* iarr_dT_MI
	, int* iarr_dT_ML, int* iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, fdmt_type_ * piarrOut);




