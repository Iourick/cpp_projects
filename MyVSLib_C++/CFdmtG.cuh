#pragma once

#include "Constants.h"
class CFdmtG
{
public:
	~CFdmtG();
	CFdmtG();
	CFdmtG(const  CFdmtG& R);
	CFdmtG& operator=(const CFdmtG& R);



	CFdmtG(
		const float Fmin
		, const float Fmax
		, int nchan // quant channels/rows of input image, including consisting of zeroes
		, const int cols
		, int imaxDt // quantity of rows of output image
	);
	//------------------------------------------------------------------------------------------
	int m_nchan; // quant channels/rows of input image, including consisting of zeroes	
	int m_cols;  // quant cols of input image (time axe)
	float m_Fmin;
	float m_Fmax;
	int m_imaxDt; // quantity of rows of output image
	// configuration params:
	// 1. quantity of iterations:
	int m_iNumIter; 
	// 2. m_parrQuantMtrx -  allocated on GPU. In this array of length (m_iNumIter +1) we will store the numbers of submatices for each iteration, including 
	// initialization. So m_parrQuantMtrx[0] = m_nchan,.. ,m_parrQuantMtrx[m_iNumIter] = 1	
	int* m_parrQuantMtrx;
	// 3.m_pparrRowsCumSum -  allocated on CPU. In this array of pointers of length m_iNumIter +1 we will store the numbers 
	// of the beginning of each submatrix for each iteration. 
	// For i =  0, the array m_pparrRowsCumSum[0]  is as follows:
	//  length = (m_parrQuantMtrx[0] +1); m_pparrRowsCumSum[0][0] = 0,m_pparrRowsCumSum[0][1] = ideltaT+1, .. ,m_pparrRowsCumSum[0][m_parrQuantMtrx[0] ] = (ideltaT+1) * m_nchan
	// For i =  1, the array m_pparrRowsCumSum[1]  is as follows:
	//  length = (m_parrQuantMtrx[1] +1); m_pparrRowsCumSum[1][0] = 0,m_pparrRowsCumSum[1][1] = quantRow0,m_pparrRowsCumSum[1][2] =m_pparrRowsCumSum[1][1] + quantRow1, .. 
	// arrays  m_pparrRowsCumSum[i] -  allocated on GPU.
	int** m_pparrRowsCumSum;
	// 4. m_pparrFreq -  allocated on CPU. In this array of pointers of length (m_iNumIter +1) we will store the bounding frequancies for chanels for each iterartion
	// including initialization. 	
	// For initialization, i =  0, the array m_pparrFreq[0]  is as follows:
	//  length = (m_parrQuantMtrx[0] +1); m_pparrFreq[0][0] = m_Fmin, m_pparrFreq[0][1] = m_Fmin +dF, m_pparrFreq[0][2] = m_Fmin +dF *2
	// , .. ,m_pparrFreq[0][m_parrQuantMtrx[0]] = m_Fmax
	// For first iteration,  i =  1, the array m_parrQuantMtrx1]  is as follows:
	//  length = (m_parrQuantMtrx[1] +1); m_parrQuantMtrx[1][0] = m_Fmin, ...,m_pparrFreq[0][m_parrQuantMtrx[0]] = m_Fmax
	// arrays m_parrQuantMtrx[i] -  allocated on GPU.
	float** m_pparrFreq;

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
	// m_parrQuantMtrxHost and m_parrMaxQuantRowsHostare allocated on CPU
	// m_parrQuantMtrxHost - CPU analogue of m_parrQuantMtrx
	// m_parrMaxQuantRowsHost has length (m_iNumIter +1) 
	// we will store in this array maximal quantity of rows of submatrices for each iteration, including initialization.
	// So, m_parrMaxQuantRowsHost[i] = pparrRowsCumSum[i][1]
	int* m_parrQuantMtrxHost;
	int* m_parrMaxQuantRowsHost;

	/*int16_t* m_parr_j0;
	int16_t* m_parr_j1;*/


	void process_image(fdmt_type_* d_parrImage       // on-device input image	
		, fdmt_type_* u_parrImOut	// OUTPUT image
		, const bool b_ones
		);

	

	static  unsigned int calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
		, const float val_DM_Max, const int nchan);

	int  calc_quant_iterations();

	int calc_deltaT(const float f0, const float f1);

	void fncFdmtIterationC(fdmt_type_* p0, int& quantSubMtrx, int* iarrQntSubmtrxRows, float* arrFreq
		, fdmt_type_* p1, const bool b_ones);

	void calcNextStateConfig(const int QuantMtrx, const int* IArrSubmtrxRows, const float* ARrFreq
		, int& quantMtrx, int* iarrSubmtrxRows, float* arrFreq);

	size_t calc_size_input();
	
	//---------------------
	size_t calc_size_output();

	size_t calcSizeAuxBuff_fdmt_();

	void  create_config(int**& pparrRowsCumSum, float**& pparrFreq, int** pparrQuantMtrx, int* piNumIter);

	int  calc_quant_iterations_and_lengthSubMtrxArray(int** pparrLength);

};
void calcCumSum(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum);

__global__
void kernel_init_fdmt0(fdmt_type_* d_parrImg, const int &IImgrows, const int *IImgcols
	, const int &IDeltaT, fdmt_type_* d_parrOut, const bool b_ones);




inline int calc_IDeltaT(const int IImgrows, const float VAlFmin, const float VAlFmax, const int IMaxDT)
{
	const float VAl_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
	return int(ceil(((float)IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin) - 1. / ((VAlFmin + VAl_dF) * (VAlFmin + VAl_dF)))
		/ (1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax))));
}

void calcCumSum_(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum);

unsigned long long ceil_power2__(const unsigned long long n);

__device__
double fnc_delay(const float fmin, const float fmax);

__global__
void kernel_fdmtIter_v1(fdmt_type_* d_parrInp, const int *cols, int& quantSubMtrx, int* iarrCumSum, float* arrFreq
	, int& quantSubMtrxCur, int* iarrCumSumCur, float* arrFreqCur
	, fdmt_type_* d_parrOut);

size_t   calcSizeAuxBuff_fdmt(const unsigned int IImrows, const unsigned int IImgcols, const  float VAlFmin
	, const  float VAlFmax, const int IMaxDT);

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
