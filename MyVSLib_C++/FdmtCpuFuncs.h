#pragma once
//-------------------------------------------------------------------------
void fncFdmt_cpu_v0(int* piarrImg, const int iImgrows
	, const int iImgcols, const float f_min
	, const  float f_max, const int imaxDT, int* piarrOut);

void fnc_init_cpu(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrOut);

void fncFdmtIteration_cpu(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1);


void create_2d_arrays_cpu(const int IDim0, const int IDim1
	, float* arr_val0, float* arr_val1, int* iarr_deltaTLocal
	, int* iarr_dT_middle_index, int* iarr_dT_middle_larger
	, int* iarr_dT_rest_index);


void shift_and_sum_cpu_v1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

void shift_and_sum_cpu(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut);

//void fncCalcDimensionsOfOutputArrays(std::vector<int>* pivctOutDim0, std::vector<int>* pivctOutDim1
//	, std::vector<int>* pivctOutDim2, const int IDim0, const int IDim1
//	, const int IDim2, const int IMaxDT, const float VAlFmin
//	, const float VAlFmax);


