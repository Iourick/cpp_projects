#pragma once

#include "Constants.h"
#include "FdmtB.h"
class CFdmtGpu:public CFdmtB
{
public:
	~CFdmtGpu();
	CFdmtGpu();
	CFdmtGpu(const  CFdmtGpu& R);
	CFdmtGpu& operator=(const CFdmtGpu& R);
	CFdmtGpu(
		const float Fmin
		, const float Fmax
		, int nchan // quant channels/rows of input image, including consisting of zeroes
		, const int cols
		, int imaxDt // quantity of rows of output image
	);
	

	// 5. buffers m_arrOut0, m_arrOut1-  allocated on GPU . In this arrays we will store input- output buffers for iterations,
	// in order to save running time for memory allocation on GPU:
	fdmt_type_* m_arrOut0;
	fdmt_type_* m_arrOut1;
	// 6. m_lenSt0, m_lenSt1 -length of m_arrOut0 and m_arrOut1respectively
	// m_lenSt0 = (pparrRowsCumSum[0])[m_parrQuantMtrxHost[0]] * m_cols;
	// m_lenSt1 = (pparrRowsCumSum[1])[m_parrQuantMtrxHost[1]] * m_cols;
	int m_lenSt0;
	int m_lenSt1;
	
	// 7. These members we need to implement to optimize kernel's managing.
	// m_parrQuantMtrxHost and m_parrMaxQuantRowsHost are allocated on CPU
	// m_parrQuantMtrxHost - CPU analogue of m_parrQuantMtrx
	// m_parrMaxQuantRowsHost has length (m_iNumIter +1) 
	// we will store in this array maximal quantity of rows of submatrices for each iteration, including initialization.
	// So, m_parrMaxQuantRowsHost[i] = pparrRowsCumSum[i][1]
	//int* m_parrQuantMtrxHost;
	int* m_parrMaxQuantRowsHost;

	/*int16_t* m_parr_j0;
	int16_t* m_parr_j1;*/
	// configuration params on device:
	int** m_pparrRowsCumSum_d;
	float** m_pparrFreq_d;
	


	void process_image(fdmt_type_* d_parrImage       // on-device input image	
		, fdmt_type_* u_parrImOut	// OUTPUT image
		, const bool b_ones
		);		

	virtual size_t calcSizeAuxBuff_fdmt_();	

};

__global__
void kernel_init_fdmt0(fdmt_type_* d_parrImg, const int &IImgrows, const int *IImgcols
	, const int &IDeltaT, fdmt_type_* d_parrOut, const bool b_ones);

__device__
double fnc_delay(const float fmin, const float fmax);

__global__
void kernel_fdmtIter_v1(fdmt_type_* d_parrInp, const int *cols, int& quantSubMtrx, int* iarrCumSum, float* arrFreq
	, int& quantSubMtrxCur, int* iarrCumSumCur, float* arrFreqCur
	, fdmt_type_* d_parrOut);


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
//-----------------------------------------------------------------------



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
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal);

__global__
void kernel_2d_arrays(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_iarr_dT_rest_index);
