#pragma once

#include "Constants.h"
class CFdmtU
{
public:
	CFdmtU();
	CFdmtU(const  CFdmtU& R);
	CFdmtU& operator=(const CFdmtU& R);
	CFdmtU(
		const float Fmin
		, const float Fmax
		, const  int nchan_act
		, const int cols
		, const int imaxDt
	);

	CFdmtU(
		const float Fmin
		, const float Fmax
		, const  int nchan_act
		, const int cols
		, const float pulse_length
		, const float d_max
		, const int lensft
	);


	int m_nchan; // quant channels/rows of input image, including consisting of zeroes
	int m_nchan_act; // quant of actual nonzeroed channels, numbers from 0 till m_nchan_act-1
	int m_cols;  // quant cols of input image (time axe)
	float m_Fmin;
	float m_Fmax;
	int m_imaxDt; // quantity of rows of output image

	void process_image(fdmt_type_* d_parrImage       // on-device input image
		, void* pAuxBuff_fdmt
		, fdmt_type_* u_parrImOut	// OUTPUT image
		, const bool b_ones
	);

	size_t calcSizeAuxBuff_fdmt_();

	size_t calc_size_input();

	size_t calc_size_output();

	static  unsigned int calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
		, const float val_DM_Max, const int nchan);

};
void fncFdmtIteration(fdmt_type_* d_parrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, fdmt_type_* d_parrOut, int& iOutPutDim0, int& iOutPutDim1);

__global__
void kernel3D_Main_012_v1(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, fdmt_type_* d_parrOut);

__global__
void kernel3D_Main_012_v2(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1, fdmt_type_* d_parrOut);

__global__
void kernel3D_Main_012_v3(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1, fdmt_type_* d_parrOut);

__host__ __device__
void calc3AuxillaryVars(int& ideltaTLocal, int& i_dT, int& iF, float& val0
	, float& val1, int& idT_middle_index, int& idT_middle_larger, int& idT_rest_index);

__global__
void kernel_shift_and_sum(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, fdmt_type_* d_parrOut);

__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal);

__global__
void kernel_2d_arrays(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_iarr_dT_rest_index);

__global__
void kernel_init_fdmt0(fdmt_type_* d_parrImg, const int IImgrows, const int IImgrows_act, const int IImgcols
	, const int IDeltaT, fdmt_type_* d_parrOut, const bool b_ones);

inline int calc_IDeltaT(const int IImgrows, const float VAlFmin, const float VAlFmax, const int IMaxDT)
{
	const float VAl_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
	return int(ceil(((float)IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin) - 1. / ((VAlFmin + VAl_dF) * (VAlFmin + VAl_dF)))
		/ (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax))));
}

size_t   calcSizeAuxBuff_fdmt(const unsigned int IImrows, const unsigned int IImgcols, const  float VAlFmin
	, const  float VAlFmax, const int IMaxDT);







