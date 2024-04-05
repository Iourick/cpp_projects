#include "FdmtU.cuh"

#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <chrono>
#include "npy.hpp"
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "yr_cart.h"

CFdmtU::CFdmtU()
{
	m_Fmin = 0.;
	m_Fmax = 0.;
	m_nchan = 0;
	m_nchan_act = 0;
	m_cols = 0;
	m_imaxDt = 0;
}
//-----------------------------------------------------------

CFdmtU::CFdmtU(const  CFdmtU& R)
{
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_nchan = R.m_nchan;
	m_nchan_act = R.m_nchan_act;
	m_cols = R.m_cols;
	m_imaxDt = R.m_imaxDt;
}
//-------------------------------------------------------------------

CFdmtU& CFdmtU::operator=(const CFdmtU& R)
{
	if (this == &R)
	{
		return *this;
	}
	m_Fmin = R.m_Fmin;
	m_Fmax = R.m_Fmax;
	m_nchan = R.m_nchan;
	m_nchan_act = R.m_nchan_act;
	m_cols = R.m_cols;
	m_imaxDt = R.m_imaxDt;
	return *this;
}
//------------------------------------------------------------------
CFdmtU::CFdmtU(
	const float Fmin
	, const float Fmax
	, const  int nchan_act		
	, const int cols
	, const int imaxDt
)
{
	m_Fmin = Fmin;
	m_Fmax = Fmax;
	m_nchan = ceil_power2_(nchan_act);
	m_nchan_act = nchan_act;
	m_cols = cols;
	m_imaxDt = imaxDt;

}
//---------------------------------------------------
CFdmtU::CFdmtU(
	const float Fmin
	, const float Fmax
	, const  int nchan_act
	, const int cols
	, const float pulse_length
	, const float d_max
	, const int lensft
)
{
	m_nchan_act = nchan_act;
	
	m_nchan = ceil_power2_(nchan_act);

	// calculation parameters for FDMT
	m_Fmin = Fmin;
	m_Fmax = m_Fmin + (Fmax - Fmin) * ((float)m_nchan) / ((float)m_nchan_act);	
	m_cols = cols;
	m_imaxDt =  calc_MaxDT(m_Fmin, m_Fmax, pulse_length, d_max, m_nchan);
}

//----------------------------------------------------
void CFdmtU::process_image(fdmt_type_* d_parrImage       // on-device input image
	, void* pAuxBuff_fdmt
	, fdmt_type_* u_parrImOut	// OUTPUT image
	, const bool b_ones
)
{
	const int IDeltaT = calc_IDeltaT(m_nchan, m_Fmin, m_Fmax, m_imaxDt);
	// installation of pointers	
	char* charPtr = (char*)pAuxBuff_fdmt;
	fdmt_type_* d_parrState0 = (fdmt_type_*)pAuxBuff_fdmt;		// on-device auxiliary memory buffer
	size_t sz0 = m_nchan * (IDeltaT + 1) * m_cols * sizeof(fdmt_type_);

	fdmt_type_* d_parrState1 = (fdmt_type_*)(charPtr + sz0);
	size_t sz1 = 2 * sz0;
	float* d_arr_val0 = (float*)(charPtr + sz1);
	size_t sz2 = sz1 + m_nchan / 2 * sizeof(float);
	float* d_arr_val1 = (float*)(charPtr + sz2);
	size_t sz3 = sz2 + m_nchan / 2 * sizeof(float);
	int* d_arr_deltaTLocal = (int*)(charPtr + sz3);
	size_t sz4 = sz3 + m_nchan / 2 * sizeof(int);
	int* d_arr_dT_MI = (int*)(charPtr + sz4);
	size_t sz5 = sz4 + m_nchan * (IDeltaT + 1) * sizeof(int);
	int* d_arr_dT_ML = (int*)(charPtr + sz5);
	size_t sz6 = sz5 + m_nchan * (IDeltaT + 1) * sizeof(int);
	int* d_arr_dT_RI = (int*)(charPtr + sz6);
	//int itempii = (char*)d_arr_dT_RI - (char*)d_parrState0 + m_nchan * (IDeltaT + 1) * sizeof(int);
	// 1. quant iteration's calculation
	const int I_F = (int)(log2((double)(m_nchan)));
	// 2. temp variables calculations
	const float VAl_dF = (m_Fmax - m_Fmin) / ((float)(m_nchan));


	auto start = std::chrono::high_resolution_clock::now();
	const dim3 blockSize = dim3(1024, 1);
	const dim3 gridSize = dim3((m_cols + blockSize.x - 1) / blockSize.x, m_nchan);
	kernel_init_fdmt0 << < gridSize, blockSize >> > (d_parrImage, m_nchan, m_nchan_act, m_cols, IDeltaT, d_parrState0, b_ones);
	cudaDeviceSynchronize();



	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by function kernel_init_fdmt0: " << duration.count() << " microseconds" << std::endl;


	/*float* parr1 = (float*)malloc(m_nchan * m_cols*(1 + IDeltaT) * sizeof(float));
	cudaMemcpy(parr1, d_parrState0, m_nchan * m_cols * (1 + IDeltaT) * sizeof(float)
		, cudaMemcpyDeviceToHost);
	std::vector<float> v7(parr1, parr1 + m_nchan * m_cols * (1 + IDeltaT) );

	std::array<long unsigned, 3> leshape127 { m_nchan, m_cols, 1 + IDeltaT };

	npy::SaveArrayAsNumpy("gpu_init.npy", false, leshape127.size(), leshape127.data(), v7);
	free(parr1);
	int ii = 0;*/

	// !1


	// 2.pointers initialization
	fdmt_type_* d_p0 = d_parrState0;
	fdmt_type_* d_p1 = d_parrState1;
	// 2!	

	// 
	int iInp0 = m_nchan;
	int iInp1 = IDeltaT + 1;

	int iOut0 = 0, iOut1 = 0, iOut2 = 0;\

	/*	float* parr = (float*)malloc(iInp0 * iInp1 * m_cols * sizeof(float));
	cudaMemcpy(parr, d_p0, iInp0 * iInp1 * m_cols * sizeof(float)
		, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	int hh = 0;
	delete[]parr;*/

	// 3. iterations
	auto start2 = clock();
	for (int iit = 1; iit < (I_F + 1); ++iit)
	{
		fncFdmtIteration(d_p0, VAl_dF, iInp0, iInp1
			, m_cols, m_imaxDt, m_Fmin
			, m_Fmax, iit, d_arr_val0
			, d_arr_val1, d_arr_deltaTLocal
			, d_arr_dT_MI, d_arr_dT_ML, d_arr_dT_RI
			, d_p1, iOut0, iOut1);

		/*float* parr1 = (float*)malloc(iOut0 * iOut1 * m_cols  * sizeof(float));
		cudaMemcpy(parr1, d_p1, iOut0 * iOut1 * m_cols * sizeof(float)
			, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		int uu = 0;
		delete[]parr1;*/

		if (iit == I_F)
		{
			break;
		}
		// exchange order of pointers
		fdmt_type_* d_pt = d_p0;
		d_p0 = d_p1;
		d_p1 = d_pt;
		iInp0 = iOut0;
		iInp1 = iOut1;
		if (iit == I_F - 1)
		{
			d_p1 = u_parrImOut;
		}
		// !
	}
	auto end2 = clock();
	auto duration2 = double(end2 - start2) / CLOCKS_PER_SEC;
	//std::cout << "Time taken by iterations: " << duration2 << " seconds" << std::endl;
	// ! 3

}
//----------------------------------------

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
void fncFdmtIteration(fdmt_type_* d_parrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float* d_arr_val0
	, float* d_arr_val1, int* d_iarr_deltaTLocal
	, int* d_iarr_dT_MI, int* d_iarr_dT_ML, int* d_iarr_dT_RI
	, fdmt_type_* d_parrOut, int& iOutPutDim0, int& iOutPutDim1)
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
	/*int* parr = (int*)malloc(iOutPutDim0 * sizeof(int));
	cudaMemcpy(parr, d_iarr_deltaTLocal, iOutPutDim0 * sizeof(int)
		, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	delete[]parr;*/

	// 12. calculating second 3 auxillary 2 dim arrays
	int quantEl = iOutPutDim0 * iOutPutDim1;
	threadsPerBlock = 256;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;

	kernel_2d_arrays << < numberOfBlocks, threadsPerBlock >> > (iOutPutDim0
		, iOutPutDim1, d_arr_val0, d_arr_val1, d_iarr_deltaTLocal
		, d_iarr_dT_MI, d_iarr_dT_ML
		, d_iarr_dT_RI);
	cudaDeviceSynchronize();

	/*int* parr = (int*)malloc(quantEl * sizeof(int));
	cudaMemcpy(parr, d_iarr_dT_RI, quantEl * sizeof(int)
		, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	delete[]parr;*/

	// !11

	// 13.
	// 2024 x 2024 takes 9.4 millisec
	/*quantEl = iOutPutDim0 * iOutPutDim1 * IDim2;
	numberOfBlocks = (quantEl + threadsPerBlock - 1) / threadsPerBlock;
	kernel_shift_and_sum << <numberOfBlocks, threadsPerBlock >> > (d_piarrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_piarrOut);
		cudaDeviceSynchronize();*/

		// 2024 x 2024 takes 8.65 millisec
	const dim3 blockSize = dim3(1024, 1, 1);
	const dim3 gridSize = dim3((IDim2 + blockSize.x - 1) / blockSize.x, iOutPutDim1, iOutPutDim0);
	// !!!
	kernel3D_Main_012_v1 << <gridSize, blockSize >> > (d_parrInp
		, IDim0, IDim1, IDim2, d_iarr_deltaTLocal, d_iarr_dT_MI, d_iarr_dT_ML, d_iarr_dT_RI
		, iOutPutDim0, iOutPutDim1, d_parrOut);
	cudaDeviceSynchronize(); 

	//float* parr = (float*)malloc(iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(float));
	//cudaMemcpy(parr, d_parrOut, iOutPutDim0* iOutPutDim1 * IDim2 *sizeof(float)
	//	, cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	//delete[]parr;
	// 2024 x 2024 takes 8.43 millisec
	/*size_t smemsize = 40;
	kernel3D_Main_012_v2 << <gridSize, blockSize, smemsize >> > (d_parrInp, IDim0, IDim1
	, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1
	, iOutPutDim0, iOutPutDim1, d_parrOut);
	cudaDeviceSynchronize();*/


	//// 2024 x 2024 takes 8.55 millisec
	///*size_t smemsize = 24;
	//kernel3D_Main_012_v3 << <gridSize, blockSize, smemsize >> > (d_parrInp, IDim0, IDim1
	//	, IDim2, d_iarr_deltaTLocal, d_arr_val0, d_arr_val1
	//	, iOutPutDim0, iOutPutDim1, d_parrOut);
	//cudaDeviceSynchronize();*/

}
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
//--------------------------------------------------------------------------------------
__global__
void kernel_init_fdmt0(fdmt_type_* d_parrImg, const int IImgrows, const int IImgrows_act, const int IImgcols
	, const int IDeltaT, fdmt_type_* d_parrOut, const bool b_ones)
{
	int i_F = blockIdx.y;
	int numOutElemInRow = blockIdx.x * blockDim.x + threadIdx.x;
	if (numOutElemInRow >= IImgcols)
	{
		return;
	}
	int numOutElemPos = i_F * (IDeltaT + 1) * IImgcols + numOutElemInRow;
	int numInpElemPos = i_F * IImgcols + numOutElemInRow;

	if (i_F >= IImgrows_act)
	{
		d_parrOut[numOutElemPos] = 0;
	}
	else
	{
		fdmt_type_ itemp = (b_ones) ? (fdmt_type_)1 : d_parrImg[numInpElemPos];
		d_parrOut[numOutElemPos] = itemp;
		for (int i_dT = 1; i_dT < (1 + IDeltaT); ++i_dT)
		{
			numOutElemPos += IImgcols;
			if (i_dT <= numOutElemInRow)
			{
				float  val = (b_ones) ? 1.0 : ((float)d_parrImg[i_F * IImgcols + numOutElemInRow - i_dT]);
				itemp = itemp + (fdmt_type_)(val);
				d_parrOut[numOutElemPos] = itemp;
			}

			else
			{
				d_parrOut[numOutElemPos] = 0;
			}
		}
	}


	//fdmt_type_  itemp = (fdmt_type_ )0;
	//if (i_F < IImgrows_act)
	//{
	//	itemp = (b_ones) ? (fdmt_type_)1 : d_parrImg[numInpElemPos];
	//}
	//
	//d_parrOut[numOutElemPos] = itemp;

	//// old variant
	//for (int i_dT = 1; i_dT < (1 + IDeltaT); ++i_dT)
	//{
	//	numOutElemPos += IImgcols;
	//	if (i_dT <= numOutElemInRow)
	//	{
	//		float  val = (b_ones) ? 1.0 : ((float)d_parrImg[i_F * IImgcols + numOutElemInRow - i_dT]);
	//		itemp = itemp + (fdmt_type_)(val);
	//		d_parrOut[numOutElemPos] = itemp;
	//	}

	//	else
	//	{
	//		d_parrOut[numOutElemPos] = 0;
	//	}
	//}	
}
//--------------------------------------------------------------------------------------


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

//--------------------------------------------------------------------------------------


size_t CFdmtU::calcSizeAuxBuff_fdmt_()
{
	const int  IDeltaT = calc_IDeltaT(m_nchan, m_Fmin, m_Fmax, m_imaxDt);

	// 1.  to 2 State arrays
	size_t szTwoStates = 2 * m_nchan * (IDeltaT + 1) * m_cols * sizeof(fdmt_type_);

	//  2. to device  auxiliary arrays
	size_t sxAux = m_nchan * sizeof(float) + m_nchan / 2 * sizeof(int) + 3 * m_nchan * (IDeltaT + 1) * sizeof(int);
	return szTwoStates + sxAux;
}
//---------------------
size_t CFdmtU::calc_size_input()
{
	return m_cols * m_nchan * sizeof(fdmt_type_ );
}
//---------------------
size_t CFdmtU::calc_size_output()
{
	return m_cols * m_imaxDt * sizeof(fdmt_type_);
}
//-----------------------------------------------------------------
unsigned int CFdmtU::calc_MaxDT(const float val_fmin_MHz, const float val_fmax_MHz, const float length_of_pulse
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


