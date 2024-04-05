#include "FdmtCu0.cuh"
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "FdmtCu0.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator

#include <algorithm> 

#include <chrono>
#include "npy.hpp"

//#include "FdmtCuKeith.cuh"
using namespace std;

//-------------------------------------------------------------------------

void fncFdmt_cu_v0(int * piarrImage // input image
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
	, const  float VAlFmin
	, const  float VAlFmax
	, const int IMaxDT
	, int* u_piarrImOut			// OUTPUT image, dim = IDeltaT x IImgcols
	)
{
	cudaMemcpy(d_piarrImage, piarrImage, IImgcols * IImgrows * sizeof(int), cudaMemcpyHostToDevice);
	// 1. call initialization func	

	auto start = std::chrono::high_resolution_clock::now();
	const dim3 blockSize = dim3(1024, 1);
	const dim3 gridSize = dim3((IImgcols + blockSize.x - 1) / blockSize.x, IImgrows);
	kernel_init_fdmt0 << < gridSize, blockSize >> > (d_piarrImage, IImgrows, IImgcols, IDeltaT, d_piarrState0);
	auto end = std::chrono::high_resolution_clock::now();	
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);	
	//std::cout << "Time taken by function kernel_init_fdmt0: " << duration.count() << " microseconds" << std::endl;

	// !1
	
	
	// 2.pointers initialization
	int *d_p0 = d_piarrState0;
	int* d_p1 = d_piarrState1;
	// 2!	

	// 
	int iInp0 = IImgrows;
	int iInp1 = IDeltaT + 1;

	int iOut0 = 0, iOut1 = 0, iOut2 = 0;

	// 3. iterations
	auto start2 = clock();
	for (int iit = 1; iit < (I_F + 1); ++iit)
	{
		fncFdmtIteration(d_p0, VAl_dF, iInp0, iInp1
			, IImgcols, IMaxDT, VAlFmin
			, VAlFmax, iit, d_arr_val0
			, d_arr_val1, d_arr_deltaTLocal
			, d_arr_dT_MI, d_arr_dT_ML, d_arr_dT_RI
			, d_p1, iOut0, iOut1);
		if (iit == I_F)
		{
			break;
		}
		// exchange order of pointers
		int* d_pt = d_p0;
		d_p0 = d_p1;
		d_p1 = d_pt;
		iInp0 = iOut0;
		iInp1 = iOut1;
		if (iit == I_F -1)
		{
			d_p1 = u_piarrImOut;
		}
		// !
	}
	auto end2 = clock();
	auto duration2 = double(end2 - start2) / CLOCKS_PER_SEC;
	//std::cout << "Time taken by iterations: " << duration2 << " seconds" << std::endl;
	// ! 3


}

//--------------------------------------------------------------------------------------
//    Input :
//    Input - 3d array, with dimensions[N_f, N_d0, Nt]
//    f_min, f_max - are the base - band begin and end frequencies.
//    The frequencies can be entered in both MHz and GHz, units are factored out in all uses.
//    maxDT - the maximal delay(in time bins) of the maximal dispersion.
//    Appears in the paper as N_{\Delta}
//A typical input is maxDT = N_f
//dataType - To naively use FFT, one must use floating point types.
//Due to casting, use either complex64 or complex128.
//iteration num - Algorithm works in log2(Nf) iterations, each iteration changes all the sizes(like in FFT)
//Output:
//3d array, with dimensions[N_f / 2, N_d1, Nt]
//    where N_d1 is the maximal number of bins the dispersion curve travels at one output frequency band
//
//    For details, see algorithm 1 in Zackay & Ofek(2014)
// F,T = Image.shape 
// d_piarrInp имеет  размерности IDim0, IDim1,IDim2
// IDim0: this is iImgrows - quantity of rows of input power image, this is F
// IDim1: changes 
// IDim2: this is iImgcols - quantity of cols of input power image, this is T 
void fncFdmtIteration(int* d_piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, int* d_piarrOut, int& iOutPutDim0, int& iOutPutDim1)
{

	float valDeltaF = pow(2., ITerNum) * val_dF;
	float temp0 = 1. / (VAlFmin * VAlFmin) -
		1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));

	const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax);

	int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
	iOutPutDim1 = ideltaT + 1;
	iOutPutDim0 = IDim0 / 2;


	// set zeros in output array
	//cudaMemset(d_piarrOut, 0, iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(int));
	// !

	float val_correction = 0;
	if (ITerNum > 0)
	{
		val_correction = val_dF / 2.;
	}


	// 9. auxiliary constants initialization
	const float VAlC2 = (VAlFmax - VAlFmin) / ((float)(iOutPutDim0));
	// !9	

	// 10. calculating first 3 auxillary 1 dim arrays
	int threadsPerBlock = 1024;
	int numberOfBlocks = (iOutPutDim0 + threadsPerBlock - 1) / threadsPerBlock;

	create_auxillary_1d_arrays << <numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
		, IMaxDT, VAlTemp1, VAlC2, VAlFmin, val_correction
		, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal);
	cudaDeviceSynchronize();

	// !10

	 

	// 12. calculating second 3 auxillary 2 dim arrays
	int quantEl = iOutPutDim0 * iOutPutDim1;
	threadsPerBlock = 256;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;

	kernel_2d_arrays << < numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
		, iOutPutDim1, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal
		, d_iarr_dT_MI, d_iarr_dT_ML
		, d_iarr_dT_RI);
	cudaDeviceSynchronize();

	// !11

	// 13.
	// 2024 x 2024 takes 9.4 millisec
	/*quantEl = iOutPutDim0 * iOutPutDim1 * IDim2;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
	kernel_shift_and_sum << <numberOfBlocks, threadsPerBlock >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut);*/

	// 2024 x 2024 takes 8.65 millisec
	const dim3 blockSize = dim3(1024,1, 1);
	const dim3 gridSize = dim3((IDim2 + blockSize.x - 1) / blockSize.x, iOutPutDim1, iOutPutDim0);

	kernel3D_Main_012_v1 << <gridSize, blockSize >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut); 

	// 2024 x 2024 takes 8.43 millisec
	/*size_t smemsize = 40;
	kernel3D_Main_012_v2 << <gridSize, blockSize, smemsize >> > (d_piarrInp, IDim0, IDim1
	, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1
	, iOutPutDim0, iOutPutDim1, d_piarrOut);*/
	
	
	// 2024 x 2024 takes 8.55 millisec
	/*size_t smemsize = 24;
	kernel3D_Main_012_v3 << <gridSize, blockSize, smemsize >> > (d_piarrInp, IDim0, IDim1
		, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1
		, iOutPutDim0, iOutPutDim1, d_piarrOut);
	cudaDeviceSynchronize();*/

	
	

}
//-----------------------------------------------------------------------------------------------------------------------

__global__
void kernel3D_Main_012_v1(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
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
	d_piarrOut[indElem] = d_piarrInp[2 * i_F * IDim1 * IDim2 + d_iarr_dT_MI[indAux] * IDim2 + numElemInRow];

	if (numElemInRow >= d_iarr_dT_ML[indAux])
	{
		int numRow = d_iarr_dT_RI[indAux];
		int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + numElemInRow - d_iarr_dT_ML[indAux];

		d_piarrOut[indElem] += d_piarrInp[indInpMtrx];
	}

}

//-----------------------------------------------------------------------------------------------------------------------

__global__
void kernel3D_Main_012_v2(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1, int* d_piarrOut)
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
	d_piarrOut[indElem] = d_piarrInp[shared_iarr[8] + numElemInRow];

	if (numElemInRow >= shared_iarr[4])
	{

		d_piarrOut[indElem] += d_piarrInp[shared_iarr[9] + numElemInRow];
	}

}
//-----------------------------------------------------------------------------------------------------------------------

__global__
void kernel3D_Main_012_v3(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, float* d_arr_val0, float* d_arr_val1
	, const int IOutPutDim0, const int IOutPutDim1, int* d_piarrOut)
{	
	extern __shared__ int sh_iarr[6];

	int i_F = blockIdx.z;
	int i_dT = blockIdx.y;
	int idT_middle_index, idT_middle_larger, idT_rest_index;
	calc3AuxillaryVars(d_iarr_deltaTLocal[i_F], i_dT, i_F, d_arr_val0[i_F]
		, d_arr_val1[i_F],  idT_middle_index, idT_middle_larger, idT_rest_index);
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
	d_piarrOut[indElem] = d_piarrInp[sh_iarr[3] + numElemInRow];

	if (numElemInRow >= sh_iarr[4])
	{

		d_piarrOut[indElem] += d_piarrInp[sh_iarr[5] + numElemInRow];
	}

}
//
//__shared__ int shared_iarr[8];
//
//int i_F = blockIdx.y;
//int i_dT = blockIdx.z;
//shared_iarr[0] = i_dT;
//shared_iarr[1] = d_iarr_deltaTLocal[i_F];
//shared_iarr[5] = i_F;
//shared_iarr[6] = IOutPutDim1 * IDim2;
//shared_iarr[7] = IDim1 * IDim2;
//calc3AuxillaryVars(d_iarr_deltaTLocal[i_F], i_dT, i_F, d_arr_val0[i_F]
//	, d_arr_val1[i_F], shared_iarr[2], shared_iarr[4], shared_iarr[3]);
//__syncthreads();
//
//if (shared_iarr[0] > shared_iarr[1])
//{
//	return;
//}
//int numCol = blockIdx.x * blockDim.x + threadIdx.x;
//if (numCol > IDim2)
//{
//	return;
//}
//
//int numElem = shared_iarr[0] * shared_iarr[6] + shared_iarr[5] * IDim2 + numCol;
//
//int numInpElem0 = shared_iarr[2] * shared_iarr[7] + 2 * shared_iarr[5] * IDim2 + numCol;
//
//int numInpElem1 = shared_iarr[3] * shared_iarr[7] + (1 + 2 * shared_iarr[5]) * IDim2 + numCol;
//// 	
//if (numCol >= shared_iarr[4])
//
//{
//	d_piarrOut[numElem] = d_piarrInp[numInpElem0] + d_piarrInp[numInpElem1 - shared_iarr[4]];
//}
//else
//{
//	d_piarrOut[numElem] = d_piarrInp[numInpElem0];
//}
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
void kernel_shift_and_sum(int* d_piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int* d_iarr_deltaTLocal, int* d_iarr_dT_MI
	, int* d_iarr_dT_ML, int* d_iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int* d_piarrOut)
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
	d_piarrOut[i] = d_piarrInp[2 * i_F * IDim1 * IDim2 +
		d_iarr_dT_MI[ind] * IDim2 + idx];

	if (idx >= d_iarr_dT_ML[ind])
	{
		int numRow = d_iarr_dT_RI[ind];
		int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + idx - d_iarr_dT_ML[ind];
		//atomicAdd(&d_piarrOut[i], d_piarrInp[ind]);
		d_piarrOut[i] += d_piarrInp[indInpMtrx];
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

	if (i > IDim0 * IDim1)
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
//--------------------------------------------------------------------------------------
__global__
void kernel_init_fdmt0(int* d_piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int* d_piarrOut)
{
	int i_F = blockIdx.y ;
	int numOutElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numOutElemInRow >= IImgcols)
	{
		return;
	}
	int numOutElemPos = i_F * (IDeltaT + 1) * IImgcols + numOutElemInRow;
	int numInpElemPos = i_F * IImgcols + numOutElemInRow;
	int  itemp = d_piarrImg[numInpElemPos];
	d_piarrOut[numOutElemPos] = itemp;

	/*for (int i_dT = 1; i_dT < (1 + IDeltaT); ++i_dT)
	{
		numOutElemPos += IImgcols;
		if (i_dT <= numOutElemInRow)
		{
			itemp = itemp + d_piarrImg[i_F * IImgcols + numOutElemInRow - i_dT];
			d_piarrOut[numOutElemPos] = itemp;
		}

		else
		{
			d_piarrOut[numOutElemPos] = 0;
		}
	}*/

	for (int i_dT = 1; i_dT < (1 + IDeltaT); ++i_dT)
	{
		numOutElemPos += IImgcols;
		if (i_dT <= numOutElemInRow)
		{
			itemp = itemp * i_dT + (int)(((float)d_piarrImg[i_F * IImgcols + numOutElemInRow - i_dT]) / ((float)(i_dT + 1.)));
			d_piarrOut[numOutElemPos] = itemp;
		}

		else
		{
			d_piarrOut[numOutElemPos] = 0;
		}
	}

	
}

//---------------------------------------------------------------------------

