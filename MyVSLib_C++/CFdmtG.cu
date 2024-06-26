#include "CFdmtG.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>
#include <cstdint>

#include <vector>
#include <chrono>
#include "npy.hpp"
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "yr_cart.h"
CFdmtG::~CFdmtG()
{
	if (m_iNumIter > 0)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrFreq[i])
			{
				cudaFree(m_pparrFreq[i]);
			}

			if (m_pparrRowsCumSum[i])
			{
				cudaFree(m_pparrRowsCumSum[i]);
			}

		}
		free(m_pparrFreq);
		free(m_pparrRowsCumSum);
		if (m_parrQuantMtrx)
		{
			cudaFree(m_parrQuantMtrx);
		}
	}

	if (m_arrOut0)
	{
		cudaFree(m_arrOut0);
	}

	if (m_arrOut1)
	{
		cudaFree(m_arrOut1);
	}

	if (m_parrQuantMtrxHost)
	{
		free(m_parrQuantMtrxHost);
	}

	if (m_parrMaxQuantRowsHost)
	{
		free(m_parrMaxQuantRowsHost);
	}

	/*if (m_parr_j0)
	{
		cudaFree(m_parr_j0);
	}

	if (m_parr_j1)
	{
		cudaFree(m_parr_j1);
	}*/
	
}
//---------------------------------------
CFdmtG::CFdmtG()
{
	m_Fmin = 0;
	m_Fmax = 0;
	m_nchan = 0;
	m_cols = 0;
	m_imaxDt = 0;
	m_pparrRowsCumSum = NULL;
	m_pparrFreq = NULL;
	m_parrQuantMtrx = NULL;
	m_arrOut0 = NULL;
	m_arrOut1 = NULL;
	m_lenSt0 = 0;
	m_lenSt1 = 0;
	m_iNumIter = 0;
	m_parrQuantMtrxHost = NULL;
	m_parrMaxQuantRowsHost = NULL;
	/*m_parr_j0 = NULL;
	m_parr_j1 = NULL;*/
}
//-----------------------------------------------------------

CFdmtG::CFdmtG(const  CFdmtG& R)
{
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_nchan = R.m_nchan;
	m_cols = R.m_cols;
	m_imaxDt = R.m_imaxDt;
	m_iNumIter = R.m_iNumIter;
	m_lenSt0 = R.m_lenSt0;
	m_lenSt1 = R.m_lenSt1;

	cudaMalloc(&m_parrQuantMtrx, (R.m_iNumIter + 1) * sizeof(int));
	cudaMemcpy(m_parrQuantMtrx, R.m_parrQuantMtrx, (R.m_iNumIter + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

	m_pparrFreq = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrFreq[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(float));
		cudaMemcpy(m_pparrFreq[i], R.m_pparrFreq[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	m_pparrRowsCumSum = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrRowsCumSum[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(int));
		cudaMemcpy(m_pparrRowsCumSum[i], R.m_pparrRowsCumSum[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	cudaMalloc(&m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_));
	cudaMemcpy(m_arrOut0, R.m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);

	cudaMalloc(&m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_));
	cudaMemcpy(m_arrOut1, R.m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);

	memcpy(m_parrQuantMtrxHost, R.m_parrQuantMtrxHost, (R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrMaxQuantRowsHost, R.m_parrMaxQuantRowsHost, (R.m_iNumIter + 1) * sizeof(int));

	/*cudaMalloc((void**)&m_parr_j0, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1]);
	cudaMemcpy(m_parr_j0, R.m_parr_j0, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1], cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&m_parr_j1, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1]);
	cudaMemcpy(m_parr_j1, R.m_parr_j1, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1], cudaMemcpyDeviceToDevice);*/
}
//-------------------------------------------------------------------
CFdmtG& CFdmtG::operator=(const CFdmtG& R)
{
	if (this == &R)
	{
		return *this;
	}
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_nchan = R.m_nchan;
	m_cols = R.m_cols;
	m_imaxDt = R.m_imaxDt;
	m_iNumIter = R.m_iNumIter;
	m_lenSt0 = R.m_lenSt0;
	m_lenSt1 = R.m_lenSt1;

	for (int i = 0; i < m_iNumIter; ++i)
	{
		cudaFree(m_pparrFreq[i]);
		cudaFree(m_pparrRowsCumSum[i]);
	}
	cudaFree(m_pparrFreq);
	cudaFree(m_pparrRowsCumSum);
	cudaFree(m_parrQuantMtrx);
	cudaFree(m_arrOut0);
	cudaFree(m_arrOut1);
	free(m_parrQuantMtrxHost);
	free(m_parrMaxQuantRowsHost);
	/*cudaFree(m_parr_j0);
	cudaFree(m_parr_j1);*/

	cudaMalloc(&m_parrQuantMtrx, (R.m_iNumIter + 1) * sizeof(int));
	cudaMemcpy(m_parrQuantMtrx, R.m_parrQuantMtrx, (R.m_iNumIter + 1) * sizeof(int), cudaMemcpyDeviceToDevice);

	m_pparrFreq = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrFreq[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(float));
		cudaMemcpy(m_pparrFreq[i], R.m_pparrFreq[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	m_pparrRowsCumSum = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrRowsCumSum[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(int));
		cudaMemcpy(m_pparrRowsCumSum[i], R.m_pparrRowsCumSum[i], (1 + R.m_parrQuantMtrx[i]) * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	cudaMalloc(&m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_));
	cudaMemcpy(m_arrOut0, R.m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);

	cudaMalloc(&m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_));
	cudaMemcpy(m_arrOut1, R.m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);

	memcpy(m_parrQuantMtrxHost, R.m_parrQuantMtrxHost, (R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrMaxQuantRowsHost, R.m_parrMaxQuantRowsHost, (R.m_iNumIter + 1) * sizeof(int));

	/*cudaMalloc((void**)&m_parr_j0, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1]);
	cudaMemcpy(m_parr_j0, R.m_parr_j0, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1], cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&m_parr_j1, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1]);
	cudaMemcpy(m_parr_j1, R.m_parr_j1, sizeof(int16_t) * R.m_parrQuantMtrxHost[1] * R.m_parrMaxQuantRowsHost[1], cudaMemcpyDeviceToDevice);*/
	return *this;
}

//--------------------------------------------------------------------
CFdmtG::CFdmtG(
	const float Fmin
	, const float Fmax
	, const int nchan // quant channels/rows of input image, including consisting of zeroes
	, const int cols
	, const int imaxDt // quantity of rows of output image
)
{
	m_nchan = nchan;
	m_Fmin = Fmin;
	m_Fmax = Fmax;
	m_cols = cols;
	m_imaxDt = imaxDt;
	 
	int** pparrRowsCumSum = NULL;
	float **pparrFreq = NULL;
	//int* parrQuantMtrx = NULL;
	create_config(pparrRowsCumSum, pparrFreq, &m_parrQuantMtrxHost, &m_iNumIter);

	cudaMalloc(&m_parrQuantMtrx, (m_iNumIter + 1) * sizeof(int));
	cudaMemcpy(m_parrQuantMtrx, m_parrQuantMtrxHost, (m_iNumIter + 1) * sizeof(int), cudaMemcpyHostToDevice);

	m_parrMaxQuantRowsHost = (int*)malloc((m_iNumIter + 1) * sizeof(int));
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		m_parrMaxQuantRowsHost[i] = (pparrRowsCumSum[i])[1];
	}

	
	m_pparrFreq = (float**)malloc((m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrFreq[i], (1 + m_parrQuantMtrxHost[i]) * sizeof(float));
		cudaMemcpy(m_pparrFreq[i], pparrFreq[i], (1 + m_parrQuantMtrxHost[i]) * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	m_pparrRowsCumSum = (int**)malloc((m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrRowsCumSum[i], (1 + m_parrQuantMtrxHost[i]) * sizeof(int));
		cudaMemcpy(m_pparrRowsCumSum[i], pparrRowsCumSum[i], (1 + m_parrQuantMtrxHost[i]) * sizeof(int), cudaMemcpyHostToDevice);
	}

	m_lenSt0 = (pparrRowsCumSum[0])[m_parrQuantMtrxHost[0]] * m_cols;
	m_lenSt1 = (pparrRowsCumSum[1])[m_parrQuantMtrxHost[1]] * m_cols;
	cudaMalloc(&m_arrOut0, m_lenSt0 * sizeof(fdmt_type_));
	cudaMalloc(&m_arrOut1, m_lenSt1 * sizeof(fdmt_type_));

	/*cudaMalloc((void**)&m_parr_j0, sizeof(int16_t) * m_parrQuantMtrxHost[1] * m_parrMaxQuantRowsHost[1]);
	cudaMalloc((void**)&m_parr_j1, sizeof(int16_t) * m_parrQuantMtrxHost[1] * m_parrMaxQuantRowsHost[1]);*/
	
}

//----------------------------------------------------
void CFdmtG::process_image(fdmt_type_* d_parrImage       // on-device input image	
	, fdmt_type_* u_parrImOut	// OUTPUT image
	, const bool b_ones
)
{
	
	int* d_mcols;
	cudaMalloc((void**)&d_mcols, sizeof(int));
	cudaMemcpy(d_mcols, &m_cols, sizeof(int), cudaMemcpyHostToDevice);

	auto start = std::chrono::high_resolution_clock::now();
	const dim3 blockSize = dim3(1024, 1);
	const dim3 gridSize = dim3((m_cols + blockSize.x - 1) / blockSize.x, m_nchan);
	kernel_init_fdmt0 << < gridSize, blockSize >> > (d_parrImage, m_parrQuantMtrx[0], d_mcols, m_pparrRowsCumSum[0][1], m_arrOut0, b_ones);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by function kernel_init_fdmt0: " << duration.count() << " microseconds" << std::endl;

	


	/*int* parr = (int*)malloc(m_nchan * m_cols*(1 + IDeltaT) * sizeof(int));
	cudaMemcpy(parr, m_arrOut0, m_nchan * m_cols * (1 + IDeltaT) * sizeof(int)
		, cudaMemcpyDeviceToHost);
	free(parr);*/
	
	/*std::vector<float> v7(parr1, parr1 + m_nchan * m_cols * (1 + IDeltaT));

	std::array<long unsigned, 3> leshape127 { m_nchan, m_cols, 1 + IDeltaT };

	npy::SaveArrayAsNumpy("gpu_init.npy", false, leshape127.size(), leshape127.data(), v7);
	
	int ii = 0;*/

	// !1


	// 2.pointers initialization
	fdmt_type_* d_p0 = m_arrOut0;
	fdmt_type_* d_p1 = m_arrOut1;
	// 2!	

	// 3. iterations
	
	auto start2 = clock();
	for (int iit = 1; iit < (m_iNumIter + 1); ++iit)	{
		
		const dim3 blockSize = dim3(1024, 1, 1);
		const dim3 gridSize = dim3((m_cols + blockSize.x - 1) / blockSize.x, m_parrMaxQuantRowsHost[iit], m_parrQuantMtrxHost[iit]);
		kernel_fdmtIter_v1 << < gridSize, blockSize >> > (d_p0, d_mcols, m_parrQuantMtrx[iit-1], m_pparrRowsCumSum[iit - 1], m_pparrFreq[iit - 1]
			, m_parrQuantMtrx[iit ], m_pparrRowsCumSum[iit], m_pparrFreq[iit], d_p1);
		
		cudaDeviceSynchronize();
		/*int* parr1 = (int*)malloc(m_lenSt1 * sizeof(int));
		cudaMemcpy(parr1, d_p1, m_lenSt1 * sizeof(int)	, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		int uu = 0;
		delete[]parr1;*/

		if (iit == m_iNumIter)
		{
			break;
		}
		// exchange order of pointers
		fdmt_type_* d_pt = d_p0;
		d_p0 = d_p1;
		d_p1 = d_pt;
		
		if (iit == m_iNumIter - 1)
		{
			d_p1 = u_parrImOut;
		}
		// !
	}
	auto end2 = clock();
	auto duration2 = double(end2 - start2) / CLOCKS_PER_SEC;
	//std::cout << "Time taken by iterations: " << duration2 << " seconds" << std::endl;
	// ! 3
	cudaFree(d_mcols);
}

//----------------------------------------
__global__
void kernel_fdmtIter_v1(fdmt_type_* d_parrInp, const int *cols, int& quantSubMtrx, int* iarrCumSum, float* arrFreq
	, int& quantSubMtrxCur, int* iarrCumSumCur, float* arrFreqCur
	, fdmt_type_* d_parrOut)
{
	__shared__ int shared_iarr[6];
	
	int numElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numElemInRow >= (*cols))
	{
		return;
	}
	int i_ch = blockIdx.z;	
	shared_iarr[0] = iarrCumSumCur[i_ch + 1] - iarrCumSumCur[i_ch]; //numRowsCur -  quant of rows of current output submatrix
	
	int i_row = blockIdx.y;	

	shared_iarr[1] = iarrCumSumCur[i_ch] * (*cols) + i_row * (*cols); // output row begin
	shared_iarr[2] = iarrCumSum[i_ch * 2] * (*cols); //input0 matrix begin
	shared_iarr[3] = iarrCumSum[i_ch * 2 + 1] * (*cols);//input1 matrix begin	

	shared_iarr[4] = fdividef(fnc_delay(arrFreq[2 * i_ch], arrFreq[2 * i_ch + 1]), fnc_delay(arrFreq[2 * i_ch], arrFreq[2 * i_ch + 2])) * i_row;//  coeff0* i_row; //j0
	shared_iarr[5] = fdividef(fnc_delay(arrFreq[2 * i_ch + 1], arrFreq[2 * i_ch + 2]), fnc_delay(arrFreq[2 * i_ch], arrFreq[2 * i_ch + 2])) * i_row; // j1
	///

	if (i_ch >= quantSubMtrxCur)
	{
		return;
	}
	if (i_row >= shared_iarr[0])
	{
		return;
	}
	
	fdmt_type_* pout = &d_parrOut[shared_iarr[1] + numElemInRow];

	fdmt_type_* pinp0 = &d_parrInp[shared_iarr[2]];

	if ((i_ch * 2 + 1) >= quantSubMtrx)
	{
		*pout = pinp0[i_row * (*cols) + numElemInRow];
		return;
	}	

	*pout = pinp0[shared_iarr[4] * (*cols) + numElemInRow];

	if (numElemInRow >= shared_iarr[4])
	{
		*pout += d_parrInp[shared_iarr[3] + shared_iarr[5] * (*cols) + numElemInRow - shared_iarr[4]];
	}
}
//--------------------------------------------------------------------------------------

size_t CFdmtG::calcSizeAuxBuff_fdmt_()
{
	size_t temp = 0;
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		temp += (1 + m_parrQuantMtrx[i]) * (sizeof(int) + sizeof(float));
	}
	size_t temp1 = (m_lenSt0 + m_lenSt1) * sizeof(fdmt_type_) + (m_iNumIter + 1) * (sizeof(int) + sizeof(float));
	return temp + temp1;
}
//---------------------
size_t CFdmtG::calc_size_input()
{
	return m_cols * m_nchan * sizeof(fdmt_type_);
}
//---------------------
size_t CFdmtG::calc_size_output()
{
	return m_cols * m_imaxDt * sizeof(fdmt_type_);
}
//-----------------------------------------------------------------
unsigned int CFdmtG::calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
	, const float val_DM_Max, const int nchan)
{
	float t0 = 4148.8 * (1.0 / (val_fmin_MHz * val_fmin_MHz) -
		1.0 / (val_fmax_MHz * val_fmax_MHz));
	float td = 4148.8 * val_DM_Max * (1.0 / (val_fmin_MHz * val_fmin_MHz) -
		1.0 / (val_fmax_MHz * val_fmax_MHz));
	float t_tel = 1.0E-6 / (val_fmax_MHz - val_fmin_MHz);
	float temp = length_of_pulse / t_tel;
	float val_M = ((td / t_tel) / (temp * temp));
	unsigned int ireturn = (unsigned int)(td / val_M / length_of_pulse);
	return (0 == ireturn) ? 1 : ireturn;
}
//-----------------------------------------
//------------------------------------------------------------------------------
void  CFdmtG::create_config(int**& pparrRowsCumSum, float**& pparrFreq, int** pparrQuantMtrx, int* piNumIter)
{
	// 1. calculation iterations quanttity *piNumIter and array *pparrQuantMtrx of quantity submatrices for each iteration
	// *pparrQuantMtrx  has length = *piNumIter +1
	// (*pparrQuantMtrx  has length)[0] = m_nchan , for initialization
	*piNumIter = calc_quant_iterations_and_lengthSubMtrxArray(pparrQuantMtrx);
	// 1!

	// 2. memory allocation for 2 auxillary arrays
	int* iarrQuantMtrx = *pparrQuantMtrx;

	pparrFreq = (float**)malloc((*piNumIter + 1) * sizeof(float*)); // Allocate memory for m pointers to int

	pparrRowsCumSum = (int**)malloc((*piNumIter + 1) * sizeof(int*));

	for (int i = 0; i < (*piNumIter + 1); ++i)
	{
		pparrFreq[i] = (float*)malloc((iarrQuantMtrx[i] + 1) * sizeof(float));
		pparrRowsCumSum[i] = (int*)malloc((iarrQuantMtrx[i] + 1) * sizeof(int));
	}

	// 2!

	// 3. initialization 0 step	 
	float* arrFreq = pparrFreq[0];
	//int* iarrRowsCumSum = pparrRowsCumSum[0];

	int* iarrQntSubmtrxRows = (int*)malloc(m_nchan * sizeof(int));

	int* iarrQntSubmtrxRowsCur = (int*)malloc(m_nchan * sizeof(int));

	const int ideltaT = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	for (int i = 0; i < m_nchan; ++i)
	{
		iarrQntSubmtrxRows[i] = ideltaT + 1;
		arrFreq[i] = m_Fmin + i * (m_Fmax - m_Fmin) / m_nchan;
	}
	arrFreq[m_nchan] = m_Fmax;
	calcCumSum_(iarrQntSubmtrxRows, iarrQuantMtrx[0], pparrRowsCumSum[0]);
	// 3!

	// 4. main loop. filling 2 config arrays	
	for (int i = 1; i < *piNumIter + 1; ++i)
	{
		calcNextStateConfig(iarrQuantMtrx[i - 1], iarrQntSubmtrxRows, pparrFreq[i - 1]
			, iarrQuantMtrx[i], iarrQntSubmtrxRowsCur, pparrFreq[i]);
		memcpy(iarrQntSubmtrxRows, iarrQntSubmtrxRowsCur, iarrQuantMtrx[i] * sizeof(int));
		calcCumSum_(iarrQntSubmtrxRowsCur, iarrQuantMtrx[i], pparrRowsCumSum[i]);
	}

	// 4!
	free(iarrQntSubmtrxRowsCur);
	free(iarrQntSubmtrxRows);
}
//----------------------------------------------
int  CFdmtG::calc_quant_iterations_and_lengthSubMtrxArray(int** pparrLength)
{

	*pparrLength = (int*)malloc((1 + ceil_power2__(m_nchan + 1)) * sizeof(int));

	int quantMtrx = m_nchan;
	(*pparrLength)[0] = quantMtrx;
	int* iarrQuantRows = new int[m_nchan];
	float* arrFreq = new float[m_nchan + 1];
	int ideltaT0 = calc_deltaT(m_Fmin, m_Fmin + (m_Fmax - m_Fmin) / m_nchan);
	for (int i = 0; i < m_nchan; ++i)
	{
		iarrQuantRows[i] = ideltaT0 + 1;
		arrFreq[i] = m_Fmin + i * (m_Fmax - m_Fmin) / m_nchan;
	}
	arrFreq[m_nchan] = m_Fmax;
	//std::cout << "row sum = : " << ( 1 + ideltaT0) * m_nchan << " ; Memory = " << (1 + ideltaT0) * m_nchan *m_cols<<std::endl;

	int qIter = 0;
	for (qIter = 0; qIter < 100; ++qIter)
	{
		if (quantMtrx == 1)
		{
			std::cout << "rows  = " << calc_deltaT(arrFreq[0], arrFreq[1]) + 1 << std::endl;
			break;
		}
		int isum = 0;
		for (int i = 0; i < quantMtrx / 2; ++i)
		{
			iarrQuantRows[i] = calc_deltaT(arrFreq[2 * i], arrFreq[2 * i + 2]) + 1;
			isum += iarrQuantRows[i];
			arrFreq[i + 1] = arrFreq[2 * i + 2];
		}
		int it = quantMtrx;

		if (quantMtrx % 2 == 1)
		{
			quantMtrx = quantMtrx / 2 + 1;
			iarrQuantRows[quantMtrx - 1] = iarrQuantRows[it - 1];
			isum += iarrQuantRows[quantMtrx - 1];
			arrFreq[quantMtrx] = m_Fmax;
		}
		else
		{
			quantMtrx = quantMtrx / 2;
		}
		(*pparrLength)[qIter + 1] = quantMtrx;
		//std::cout << "iter = "<< qIter << "  quant marx =    "<< quantMtrx<< "  row sum = : " << isum << "Memory = " << isum * m_cols << std::endl;
		/*for (int i = 0; i < quantMtrx; ++i)
		{
			std::cout << "iarrQuantRows[ " << i << "] = " << iarrQuantRows[i] << "  arrFreq[ " <<i<< "] = "<< arrFreq[i]  << std::endl;
		}*/
	}
	*pparrLength = (int*)realloc((*pparrLength), (qIter + 1) * sizeof(int));
	delete[]iarrQuantRows;
	delete[]arrFreq;
	return qIter;
}
//--------------------------------------------------------------------------------------
__global__
void kernel_init_fdmt0(fdmt_type_* d_parrImg, const int &IImgrows, const int *IImgcols
	, const int &IDeltaTP1, fdmt_type_* d_parrOut, const bool b_ones)
{
	int i_F = blockIdx.y;
	int numOutElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numOutElemInRow >= (*IImgcols))
	{
		return;
	}
	int numOutElemPos = i_F * IDeltaTP1 * (*IImgcols) + numOutElemInRow;
	int numInpElemPos = i_F *(* IImgcols) + numOutElemInRow;
	float  itemp = (b_ones) ? 1.0f : (float)d_parrImg[numInpElemPos];
	d_parrOut[numOutElemPos] = (fdmt_type_)itemp;

	// old variant
	for (int i_dT = 1; i_dT < IDeltaTP1; ++i_dT)
	{
		numOutElemPos += (*IImgcols);
		if (i_dT <= numOutElemInRow)
		{
			float  val = (b_ones) ? 1.0 : ((float)d_parrImg[i_F * (*IImgcols) + numOutElemInRow - i_dT]);
			itemp = fdividef(fmaf(itemp, (float)i_dT, val), (i_dT + 1));
				
			//itemp =( itemp * ((fdmt_type_)i_dT) + (fdmt_type_)(val))/((fdmt_type_)(i_dT +1));
			d_parrOut[numOutElemPos] = (fdmt_type_)itemp;
		}

		else
		{
			d_parrOut[numOutElemPos] = 0;
		}
	}

	//// new variant
	//for (int i_dT = 1; i_dT < (1 + IDeltaT); ++i_dT)
	//{
	//	numOutElemPos += IImgcols;
	//	if (i_dT <= numOutElemInRow)
	//	{
	//		float  val = (b_ones) ? 1.0 : ((float)d_parrImg[i_F * IImgcols + numOutElemInRow - i_dT]);
	//		itemp = itemp * i_dT + (fdmt_type_)(val / ((float)(i_dT + 1.)));
	//		d_parrOut[numOutElemPos] = itemp;
	//	}

	//	else
	//	{
	//		d_parrOut[numOutElemPos] = 0;
	//	}
	//}
}

//-------------------------------------------
unsigned long long ceil_power2__(const unsigned long long n)
{
	unsigned long long irez = 1;
	for (int i = 0; i < 63; ++i)
	{
		if (irez >= n)
		{
			return irez;
		}
		irez = irez << 1;
	}
	return -1;
}
//----------------------------------------
__device__
double fnc_delay(const float fmin, const float fmax)
{
	return fdividef(1.0f, fmin * fmin) - fdividef(1.0f, fmax * fmax);
	//1.0 / (fmin * fmin) - 1.0 / (fmax * fmax);
}
//-----------------------------------------//---------------------------------
int CFdmtG::calc_deltaT(const float f0, const float f1)
{
	return (int)(ceil((1.0 / (f0 * f0) - 1.0 / (f1 * f1)) / (1.0 / (m_Fmin * m_Fmin) - 1.0 / (m_Fmax * m_Fmax)) * (m_imaxDt - 1.0)));
}
//---------------------------------------------------------------------------
void CFdmtG::calcNextStateConfig(const int QuantMtrx, const int* IArrSubmtrxRows, const float* ARrFreq
	, int& quantMtrx, int* iarrSubmtrxRows, float* arrFreq)
{
	int i = 0;
	for (i = 0; i < QuantMtrx / 2; ++i)
	{
		iarrSubmtrxRows[i] = calc_deltaT(ARrFreq[2 * i], ARrFreq[2 * i + 2]) + 1;
		arrFreq[i] = ARrFreq[2 * i];
	}
	i--;
	if (QuantMtrx % 2 == 1)
	{
		quantMtrx = QuantMtrx / 2 + 1;
		arrFreq[quantMtrx - 1] = ARrFreq[2 * i + 2];
		iarrSubmtrxRows[quantMtrx - 1] = IArrSubmtrxRows[QuantMtrx - 1];
	}
	else
	{
		quantMtrx = QuantMtrx / 2;
	}
	arrFreq[quantMtrx] = m_Fmax;
}
//------------------------------------------------------------------------------------------------
void calcCumSum_(const int* iarrQntSubmtrxRows, const int quantSubMtrx, int* iarrCumSum)
{
	iarrCumSum[0] = 0;
	for (int i = 1; i < (1 + quantSubMtrx); ++i)
	{
		iarrCumSum[i] = iarrCumSum[i - 1] + iarrQntSubmtrxRows[i - 1];
	}
}
//-------------------------------------------------------
size_t   calcSizeAuxBuff_fdmt(const unsigned int IImrows, const unsigned int IImgcols, const  float VAlFmin
	, const  float VAlFmax, const int IMaxDT)
{
	const int  IDeltaT = calc_IDeltaT(IImrows, VAlFmin, VAlFmax, IMaxDT);

	// 1.  to 2 State arrays
	size_t szTwoStates = 2 * IImrows * (IDeltaT + 1) * IImgcols * sizeof(fdmt_type_);

	//  2. to device  auxiliary arrays
	size_t sxAux = IImrows * sizeof(float) + IImrows / 2 * sizeof(int) + 3 * IImrows * (IDeltaT + 1) * sizeof(int);
	return szTwoStates + sxAux;
}
//--------------------------------------------------------

//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------

__global__
void kernel3D_Main_012_v1(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, fdmt_type_* d_parrOut)
{	

	int numElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numElemInRow >= IDim2)
	{
		return;
	}
	int i_F = blockIdx.z;
	int i_dT = blockIdx.y;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int indAux = i_F * IOutPutDim1 + i_dT;
	int indElem = i_F * IOutPutDim1 * IDim2 + i_dT * IDim2 + numElemInRow;
	
	
	d_parrOut[indElem] = d_parrInp[2 * i_F * IDim1 * IDim2 + d_iarr_dT_MI[indAux] * IDim2 + numElemInRow];

	if (numElemInRow >= d_iarr_dT_ML[indAux])
	{
		int numRow = d_iarr_dT_RI[indAux];
		int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + numElemInRow - d_iarr_dT_ML[indAux];

		d_parrOut[indElem] += d_parrInp[indInpMtrx];
	}

}

//-----------------------------------------------------------------------------------------------------------------------

__global__
void kernel3D_Main_012_v2(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1, fdmt_type_* d_parrOut)
{
	extern __shared__ int shared_iarr[10];

	int i_F = blockIdx.z;
	int i_dT = blockIdx.y;
	shared_iarr[0] = i_dT;
	shared_iarr[1] = d_iarr_deltaTLocal[i_F];
	shared_iarr[5] = i_F;
	shared_iarr[6] = IOutPutDim1 * IDim2;
	shared_iarr[7] = IDim1 * IDim2;
	calc3AuxillaryVars(d_iarr_deltaTLocal[i_F], i_dT, i_F, d_arr_val0[i_F]
		, d_arr_val1[i_F], shared_iarr[2], shared_iarr[4], shared_iarr[3]);
	shared_iarr[8] = 2 * shared_iarr[5] * shared_iarr[7] + shared_iarr[2] * IDim2;
	shared_iarr[9] = (2 * i_F + 1) * IDim1 * IDim2 + shared_iarr[3] * IDim2 - shared_iarr[4];
	__syncthreads();


	int numElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (shared_iarr[0] > shared_iarr[1])
	{
		return;
	}

	if (numElemInRow >= IDim2)
	{
		return;
	}

	int indElem = shared_iarr[5] * shared_iarr[6] + shared_iarr[0] * IDim2 + numElemInRow;
	d_parrOut[indElem] = d_parrInp[shared_iarr[8] + numElemInRow];

	if (numElemInRow >= shared_iarr[4])
	{

		d_parrOut[indElem] += d_parrInp[shared_iarr[9] + numElemInRow];
	}

}
//-----------------------------------------------------------------------------------------------------------------------

__global__
void kernel3D_Main_012_v3(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1, fdmt_type_* d_parrOut)
{
	extern __shared__ int sh_iarr[6];

	int i_F = blockIdx.z;
	int i_dT = blockIdx.y;
	int idT_middle_index, idT_middle_larger, idT_rest_index;
	calc3AuxillaryVars(d_iarr_deltaTLocal[i_F], i_dT, i_F, d_arr_val0[i_F]
		, d_arr_val1[i_F], idT_middle_index, idT_middle_larger, idT_rest_index);
	sh_iarr[0] = i_dT;
	sh_iarr[1] = d_iarr_deltaTLocal[i_F];
	sh_iarr[2] = i_F * IOutPutDim1 * IDim2 + sh_iarr[0] * IDim2;
	sh_iarr[3] = 2 * i_F * IDim1 * IDim2 + idT_middle_index * IDim2;
	sh_iarr[4] = idT_middle_larger;
	sh_iarr[5] = (2 * i_F + 1) * IDim1 * IDim2 + idT_rest_index * IDim2 - idT_middle_larger;
	__syncthreads();


	int numElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (sh_iarr[0] > sh_iarr[1])
	{
		return;
	}

	if (numElemInRow >= IDim2)
	{
		return;
	}

	int indElem = sh_iarr[2] + numElemInRow;
	d_parrOut[indElem] = d_parrInp[sh_iarr[3] + numElemInRow];

	if (numElemInRow >= sh_iarr[4])
	{

		d_parrOut[indElem] += d_parrInp[sh_iarr[5] + numElemInRow];
	}

}

//--------------------------------------------------------------------------
__host__ __device__
void calc3AuxillaryVars(int& ideltaTLocal, int& i_dT, int& iF, float& val0
	, float& val1, int& idT_middle_index, int& idT_middle_larger, int& idT_rest_index)
{
	if (i_dT > ideltaTLocal)
	{
		idT_middle_index = 0;
		idT_middle_larger = 0;
		idT_rest_index = 0;
		return;
	}

	idT_middle_index = round(((float)i_dT) * val0);
	int ivalt = round(((float)i_dT) * val1);
	idT_middle_larger = ivalt;
	idT_rest_index = i_dT - ivalt;
}

//-----------------------------------------------------------------------------------------------------------------------
__global__
void kernel_shift_and_sum(fdmt_type_* d_parrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, fdmt_type_* d_parrOut)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= IOutPutDim0 * IOutPutDim1 * IDim2)
	{
		return;
	}
	int iw = IOutPutDim1 * IDim2;
	int i_F = i / iw;
	int irest = i % iw;
	int i_dT = irest / IDim2;
	if (i_dT > d_iarr_deltaTLocal[i_F])
	{
		return;
	}
	int idx = irest % IDim2;
	// claculation of bound index: 
	// arr_dT_ML[i_F, i_dT]
	// index of arr_dT_ML
	// arr_dT_ML is matrix with IOutPutDim0 rows and IOutPutDim1 cols
	int ind = i_F * IOutPutDim1 + i_dT;
	// !

	// calculation of:
	// d_Output[i_F][i_dT][idx] = d_input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
	  // calculation num row of submatix No_2 * i_F of d_piarrInp = arr_dT_MI[ind]
	d_parrOut[i] = d_parrInp[2 * i_F * IDim1 * IDim2 + d_iarr_dT_MI[ind] * IDim2 + idx];

	if (idx >= d_iarr_dT_ML[ind])
	{
		int numRow = d_iarr_dT_RI[ind];
		int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + idx - d_iarr_dT_ML[ind];
		//atomicAdd(&d_piarrOut[i], d_piarrInp[ind]);
		d_parrOut[i] += d_parrInp[indInpMtrx];
	}
}
//-----------------------------------------------------------------------------------------------------------------------
__global__
void create_auxillary_1d_arrays(const int IFjumps, const int IMaxDT, const float VAlTemp1
	, const float VAlc2, const float VAlf_min, const float VAlcorrection
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > IFjumps)
	{
		return;
	}
	float valf_start = VAlc2 * i + VAlf_min;
	float valf_end = valf_start + VAlc2;
	float valf_middle_larger = VAlc2 / 2. + valf_start + VAlcorrection;
	float valf_middle = VAlc2 / 2. + valf_start - VAlcorrection;
	float temp0 = 1. / (valf_start * valf_start) - 1. / (valf_end * valf_end);

	d_arr_val0[i] = -(1. / (valf_middle * valf_middle) - 1. / (valf_start * valf_start)) / temp0;

	d_arr_val1[i] = -(1. / (valf_middle_larger * valf_middle_larger)
		- 1. / (valf_start * valf_start)) / temp0;

	d_iarr_deltaTLocal[i] = (int)(ceil((((float)(IMaxDT)) - 1.) * temp0 / VAlTemp1));

}
//--------------------------------------------------------------------------------------
__global__
void kernel_2d_arrays(const int IDim0, const int IDim1
	, float* d_arr_val0, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_middle_index, int* d_iarr_dT_middle_larger
	, int* d_iarr_dT_rest_index)

{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= IDim0 * IDim1)
	{
		return;
	}
	int i_F = i / IDim1;
	int i_dT = i % IDim1;
	if (i_dT > (d_iarr_deltaTLocal[i_F]))
	{
		d_iarr_dT_middle_index[i] = 0;
		d_iarr_dT_middle_larger[i] = 0;
		d_iarr_dT_rest_index[i] = 0;
		return;
	}

	d_iarr_dT_middle_index[i] = round(((float)i_dT) * d_arr_val0[i_F]);
	int ivalt = round(((float)i_dT) * d_arr_val1[i_F]);
	d_iarr_dT_middle_larger[i] = ivalt;
	d_iarr_dT_rest_index[i] = i_dT - ivalt;


}



