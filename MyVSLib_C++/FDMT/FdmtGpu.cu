#include "FdmtGpu.cuh"

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


cudaError_t cudaStatus1;

CFdmtGpu::~CFdmtGpu() 
{
	printf("dextr \n");
	if (m_pparrFreq_d !=NULL)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrFreq_d[i] != NULL)
			{
				cudaFree(m_pparrFreq_d[i]);
				m_pparrFreq_d[i] = NULL;
			}			
		}
		free(m_pparrFreq_d);
		m_pparrFreq_d = NULL;
	}

	if (m_pparrRowsCumSum_d != NULL)
	{
		for (int i = 0; i < (m_iNumIter + 1); ++i)
		{
			if (m_pparrRowsCumSum_d[i] != NULL)
			{
				cudaFree(m_pparrRowsCumSum_d[i]);
				m_pparrRowsCumSum_d[i] = NULL;
			}			
		}
		free(m_pparrRowsCumSum_d);
		m_pparrRowsCumSum_d = NULL;			
	
	}

	if (m_arrOut0!= NULL)
	{
	cudaFree(m_arrOut0);
	m_arrOut0 = NULL;
	}

	if (m_arrOut1 != NULL)
	{
	cudaFree(m_arrOut1);
	m_arrOut1 = NULL;
	}

	if (m_parrMaxQuantRows_h != NULL)
	{
	free(m_parrMaxQuantRows_h);
	m_parrMaxQuantRows_h = NULL;
	}

	if (m_parrQuantMtrx_d != NULL)
	{
		cudaFree(m_parrQuantMtrx_d);
		m_parrQuantMtrx_d = NULL;
	}

	if (m_pcols_d != NULL)
	{
		cudaFree(m_pcols_d);
		m_pcols_d = NULL;
	}
	
	
}
//---------------------------------------
CFdmtGpu::CFdmtGpu():CFdmtCpu()
{	
	m_arrOut0 = NULL;
	m_arrOut1 = NULL;
	m_lenSt0 = 0;
	m_lenSt1 = 0;
	m_iNumIter = 0;	
	m_parrMaxQuantRows_h = NULL;
	m_pparrRowsCumSum_d = NULL;
	m_pparrFreq_d = NULL;
	m_parrQuantMtrx_d = NULL;
	m_pcols_d = NULL;
}
//-----------------------------------------------------------

CFdmtGpu::CFdmtGpu(const  CFdmtGpu& R) :CFdmtCpu()
{	
	m_lenSt0 = R.m_lenSt0;
	m_lenSt1 = R.m_lenSt1;	

	
	cudaMalloc((void**) & m_parrQuantMtrx_d, (1 + R.m_iNumIter) * sizeof(int));
	cudaMemcpy(m_parrQuantMtrx_d, R.m_parrQuantMtrx_d, (1 + R.m_iNumIter) * sizeof(int), cudaMemcpyDeviceToDevice);

	cudaMalloc((void**)&m_pcols_d, sizeof(int));
	cudaMemcpy(m_pcols_d, R.m_pcols_d,  sizeof(int), cudaMemcpyDeviceToDevice);
	

	m_pparrFreq_d = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrFreq_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(float));
		cudaMemcpy(m_pparrFreq_d[i], R.m_pparrFreq_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(float), cudaMemcpyDeviceToDevice);
	}	

	m_pparrRowsCumSum_d = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrRowsCumSum_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(int));
		cudaMemcpy(m_pparrRowsCumSum_d[i], R.m_pparrRowsCumSum_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	cudaMalloc(&m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_));
	cudaMemcpy(m_arrOut0, R.m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);

	cudaMalloc(&m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_));
	cudaMemcpy(m_arrOut1, R.m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);	

	m_parrMaxQuantRows_h = (int*)malloc((R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrMaxQuantRows_h, R.m_parrMaxQuantRows_h, (R.m_iNumIter + 1) * sizeof(int));
}
//-------------------------------------------------------------------
CFdmtGpu& CFdmtGpu::operator=(const CFdmtGpu& R)
{
	if (this == &R)
	{
		return *this;
	}
	CFdmtCpu:: operator= (R);
	m_lenSt0 = R.m_lenSt0;
	m_lenSt1 = R.m_lenSt1;

	if (m_parrQuantMtrx_d != NULL)
	{
		cudaFree(m_parrQuantMtrx_d);
		m_parrQuantMtrx_d = NULL;
	}
	cudaMalloc((void**)&m_parrQuantMtrx_d, (1 + R.m_iNumIter) * sizeof(int));
	cudaMemcpy(m_parrQuantMtrx_d, R.m_parrQuantMtrx_d, (1 + R.m_iNumIter) * sizeof(int), cudaMemcpyDeviceToDevice);

	if (m_pcols_d != NULL)
	{
		cudaFree(m_pcols_d);
		m_pcols_d = NULL;
	}
	cudaMalloc((void**)&m_pcols_d, sizeof(int));
	cudaMemcpy(m_pcols_d, R.m_pcols_d, sizeof(int), cudaMemcpyDeviceToDevice);

	if (m_pparrFreq_d != NULL)
	{
		for (int i = 0; i < (R.m_iNumIter + 1); ++i)
		{
			if (m_pparrFreq_d[i] != NULL)
			{
				cudaFree(m_pparrFreq_d[i]);
			}			
		}
		free(m_pparrFreq_d);
	}

	m_pparrFreq_d = (float**)malloc((R.m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrFreq_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(float));
		cudaMemcpy(m_pparrFreq_d[i], R.m_pparrFreq_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	
	if (m_pparrRowsCumSum_d != NULL)
	{
		for (int i = 0; i < m_iNumIter + 1; ++i)
		{
			if (m_pparrRowsCumSum_d[i] != NULL)
			{
				cudaFree(m_pparrRowsCumSum_d[i]);
			}			

		}
		free(m_pparrRowsCumSum_d);
	}
	m_pparrRowsCumSum_d = (int**)malloc((R.m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (R.m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrRowsCumSum_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(int));
		cudaMemcpy(m_pparrRowsCumSum_d[i], R.m_pparrRowsCumSum_d[i], (1 + R.m_parrQuantMtrx_h[i]) * sizeof(int), cudaMemcpyDeviceToDevice);
	}		
	
	
	//!
	if (m_arrOut0 != NULL)
	{
		cudaFree(m_arrOut0);
	}
	cudaMalloc(&m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_));
	//cudaMemcpy(m_arrOut0, R.m_arrOut0, R.m_lenSt0 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);

	if (m_arrOut1 != NULL)
	{
		cudaFree(m_arrOut1);
	}
	cudaMalloc(&m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_));
	//cudaMemcpy(m_arrOut1, R.m_arrOut1, R.m_lenSt1 * sizeof(fdmt_type_), cudaMemcpyDeviceToDevice);
	//!
	
	
	//!
	if (m_parrMaxQuantRows_h != NULL)
	{
		free(m_parrMaxQuantRows_h);
	}
	m_parrMaxQuantRows_h = (int*)malloc((R.m_iNumIter + 1) * sizeof(int));
	memcpy(m_parrMaxQuantRows_h, R.m_parrMaxQuantRows_h, (R.m_iNumIter + 1) * sizeof(int));

	return *this;
}

//--------------------------------------------------------------------
CFdmtGpu::CFdmtGpu(
	const float Fmin
	, const float Fmax
	, const int nchan // quant channels/rows of input image, including consisting of zeroes
	, const int cols
	, const int imaxDt // quantity of rows of output image
) : CFdmtCpu(Fmin, Fmax, nchan, cols, imaxDt)
{	
	cudaMalloc((void**)&m_parrQuantMtrx_d, (1 + m_iNumIter) * sizeof(int));
	cudaMemcpy(m_parrQuantMtrx_d, m_parrQuantMtrx_h, (1 + m_iNumIter) * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_pcols_d, sizeof(int));
	cudaMemcpy(m_pcols_d, &m_cols, sizeof(int), cudaMemcpyHostToDevice);
	

	m_parrMaxQuantRows_h = (int*)malloc((m_iNumIter + 1) * sizeof(int));
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		m_parrMaxQuantRows_h[i] = (m_pparrRowsCumSum_h[i])[1];
	}

	
	m_pparrFreq_d = (float**)malloc((m_iNumIter + 1) * sizeof(float*));
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrFreq_d[i], (1 + m_parrQuantMtrx_h[i]) * sizeof(float));
		cudaMemcpy(m_pparrFreq_d[i], m_pparrFreq_h[i], (1 + m_parrQuantMtrx_h[i]) * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	m_pparrRowsCumSum_d = (int**)malloc((m_iNumIter + 1) * sizeof(int*));
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		cudaMalloc(&m_pparrRowsCumSum_d[i], (1 + m_parrQuantMtrx_h[i]) * sizeof(int));
		cudaMemcpy(m_pparrRowsCumSum_d[i], m_pparrRowsCumSum_h[i], (1 + m_parrQuantMtrx_h[i]) * sizeof(int), cudaMemcpyHostToDevice);
	}

	m_lenSt0 = (m_pparrRowsCumSum_h[0])[m_parrQuantMtrx_h[0]] * m_cols;
	m_lenSt1 = (m_pparrRowsCumSum_h[1])[m_parrQuantMtrx_h[1]] * m_cols;
	cudaMalloc(&m_arrOut0, m_lenSt0 * sizeof(fdmt_type_));
	cudaMalloc(&m_arrOut1, m_lenSt1 * sizeof(fdmt_type_));
}

//----------------------------------------------------
void CFdmtGpu::process_image(fdmt_type_* __restrict d_parrImage       // on-device input image	
	, fdmt_type_* __restrict u_parrImOut	// OUTPUT image
	, const bool b_ones
)
{	
	auto start = std::chrono::high_resolution_clock::now();
	const dim3 blockSize = dim3(1024, 1);
	const dim3 gridSize = dim3((m_cols + blockSize.x - 1) / blockSize.x, m_nchan);
	kernel_init_fdmt0 << < gridSize, blockSize >> > (d_parrImage, m_parrQuantMtrx_d[0], m_pcols_d, m_pparrRowsCumSum_d[0][1], m_arrOut0, b_ones);
	//cudaDeviceSynchronize();

	cudaStatus1 = cudaGetLastError();
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus1));
		return ;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	//std::cout << "Time taken by function kernel_init_fdmt0: " << duration.count() << " microseconds" << std::endl;
	
	/*float* parr = (float*)malloc(m_nchan * m_cols*m_pparrRowsCumSum_h[0][1] * sizeof(float));
	cudaMemcpy(parr, m_arrOut0, m_nchan * m_cols *  sizeof(float)
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
		const dim3 gridSize = dim3((m_cols + blockSize.x - 1) / blockSize.x, m_parrMaxQuantRows_h[iit], m_parrQuantMtrx_h[iit]);
		kernel_fdmtIter_v1 << < gridSize, blockSize >> > (d_p0, m_pcols_d, m_parrQuantMtrx_d[iit-1], m_pparrRowsCumSum_d[iit - 1], m_pparrFreq_d[iit - 1]
			, m_parrQuantMtrx_d[iit ], m_pparrRowsCumSum_d[iit], m_pparrFreq_d[iit], d_p1);		
		//cudaDeviceSynchronize();
		
		cudaStatus1 = cudaGetLastError();
		if (cudaStatus1 != cudaSuccess) {
			fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus1));
			return;
		}

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
}

//----------------------------------------
__global__
void kernel_fdmtIter_v1(fdmt_type_* __restrict d_parrInp, const int *cols, int& quantSubMtrx, int* iarrCumSum, float* __restrict arrFreq
	, int& quantSubMtrxCur, int* __restrict iarrCumSumCur, float* __restrict arrFreqCur
	, fdmt_type_* __restrict d_parrOut)
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

size_t CFdmtGpu::calcSizeAuxBuff_fdmt_()
{
	size_t temp = 0;
	for (int i = 0; i < (m_iNumIter + 1); ++i)
	{
		temp += (1 + m_parrQuantMtrx_h[i]) * (sizeof(int) + sizeof(float));
	}
	size_t temp1 = (m_lenSt0 + m_lenSt1) * sizeof(fdmt_type_) + (m_iNumIter + 1) * (sizeof(int) + sizeof(float));
	return temp + temp1;
}


//--------------------------------------------------------------------------------------
__global__
void kernel_init_fdmt0(fdmt_type_* __restrict d_parrImg, const int &IImgrows, const int *IImgcols
	, const int &IDeltaTP1, fdmt_type_* __restrict d_parrOut, const bool b_ones)
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
	//printf("init");
	// old variant
	for (int i_dT = 1; i_dT < IDeltaTP1; ++i_dT)
	{
		numOutElemPos += (*IImgcols);
		if (i_dT <= numOutElemInRow)
		{
			float  val = (b_ones) ? 1.0 : ((float)d_parrImg[i_F * (*IImgcols) + numOutElemInRow - i_dT]);
			itemp = fdividef(fmaf(itemp, (float)i_dT, val), (i_dT + 1));		
			d_parrOut[numOutElemPos] = (fdmt_type_)itemp;
		}
		else
		{
			d_parrOut[numOutElemPos] = 0;
		}
	}

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
}
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------

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
//-----------------------------------------------------------------------------




