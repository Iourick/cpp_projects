#include "FdmtCpu_omp.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "npy.hpp"
#include <algorithm> 
#include <stdlib.h>

using namespace std;
//-------------------------------------------------------------------------
void fncFdmt_cpu_v0(int* piarrImgInp, const int IImgrows, const int IImgcols
	, const float VAlFmin, const  float VAlFmax, const int IMaxDT, int* piarrImgOut)
{
	// 1. quant iteration's calculation
	const int I_F = (int)(log2((double)(IImgrows)));
	// !1
	
	// 2. calc. constants
	const float val_dF = (VAlFmax - VAlFmin) / ((float)(IImgrows));
	int ideltaT = int(ceil((IMaxDT - 1.) * (1. / (VAlFmin * VAlFmin)
		- 1. / ((VAlFmin + val_dF) * (VAlFmin + val_dF)))
		/ (1. / (VAlFmin
			* VAlFmin) - 1. / (VAlFmax * VAlFmax))));
	// !2

	// 3. declare pointers 
	int*  p0 = 0;
	int*  p1 = 0;
	int*  piarrOut_0 = 0;
	int*  piarrOut_1 = 0;	
	// !3

	// 4. allocate memory 
	
	piarrOut_0 = (int*)calloc(IImgrows * (ideltaT + 1) * IImgcols, sizeof(int));
	piarrOut_1 = (int*)calloc(IImgrows * (ideltaT + 1) * IImgcols, sizeof(int));
	// !4
	

	// 5  Initialize the arrays with zeros
	
	memset( piarrOut_0, 0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
	memset( piarrOut_1, 0, IImgrows * (ideltaT + 1) * IImgcols * sizeof(int));
	// !5
	
	// 6. call initialization func
	clock_t start = clock() * 1000.;
	fnc_init_cpu( piarrImgInp, IImgrows, IImgcols, ideltaT,  piarrOut_0);
	clock_t end = clock() * 1000.;
	double duration = double(end - start) / CLOCKS_PER_SEC;
	//std::cout << "Time taken by fnc_init_cpu: " << duration << " miliseconds" << std::endl;
	// !6
	
	// 7.pointers fixing
	 p0 =  piarrOut_0;
	 p1 =  piarrOut_1;
	// 7!

	// 8. allocate memory to  auxiliary arrays
	
	float*  arr_val0 = 0;
	arr_val0 = (float*)malloc(IImgrows / 2 * sizeof(float));
	

	float*  arr_val1 = 0;
	arr_val1 = (float*)calloc(IImgrows / 2, sizeof(float));

	int*  arr_deltaTLocal = 0;
	arr_deltaTLocal = (int*)calloc(IImgrows / 2, sizeof(int));
	

	// 11. memory allocation for 3 auxillaty 2 dimensional arrays on GPU
	int*  arr_dT_MI = 0;
	arr_dT_MI = (int*)calloc(IImgrows * (ideltaT + 1), sizeof(int));
	

	int*  arr_dT_ML = 0;
	arr_dT_ML = (int*)calloc(IImgrows * (ideltaT + 1), sizeof(int));
	

	int*  arr_dT_RI = 0;
	arr_dT_RI = (int*)calloc(IImgrows * (ideltaT + 1), sizeof(int));
	
	// !8

	// 9. intialisations
	int iInp0 = IImgrows;
	int iInp1 = ideltaT + 1;

	int iOut0 = 0, iOut1 = 0, iOut2 = 0;
	// !9

	//// output in .npy:
	//int* parrinit2 = (int*)malloc(IImgrows * IImgcols * (1 + ideltaT) * sizeof(int));
	//cudaMemcpy(parrinit2,  piarrOut_0, IImgrows* IImgcols* (1 + ideltaT) * sizeof(int)
	//	, cudaMemcpyDeviceToHost);
	//std::vector<int> v2(parrinit2, parrinit2 + IImgrows * IImgcols * (1 + ideltaT));

	//std::array<long unsigned, 1> leshape122 {IImgrows* IImgcols* (1 + ideltaT)};

	//npy::SaveArrayAsNumpy("init_arr2.npy", false, leshape122.size(), leshape122.data(), v2);
	//free(parrinit2);

	// 10. calculations
	for (int iit = 1; iit < (I_F + 1); ++iit)
	{
		fncFdmtIteration_cpu( p0, val_dF, iInp0, iInp1
			, IImgcols, IMaxDT, VAlFmin
			, VAlFmax, iit,  arr_val0
			,  arr_val1,  arr_deltaTLocal
			,  arr_dT_MI,  arr_dT_ML,  arr_dT_RI
			,  p1, iOut0, iOut1);
		
		// exchange order of pointers
		int*  pt =  p0;
		 p0 =  p1;
		 p1 =  pt;
		iInp0 = iOut0;
		iInp1 = iOut1;

		
	}
	// !10

	memcpy(piarrImgOut,  p0, IImgcols * IMaxDT* sizeof(int));
	start = clock();
	free( arr_val0);
	free( arr_val1);
	free( arr_deltaTLocal);

	free( piarrOut_0);
	free( piarrOut_1);
	

	// 14. free memory
	free( arr_dT_MI);
	free( arr_dT_ML);
	free( arr_dT_RI);

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
//  piarrInp имеет  размерности IDim0, IDim1,IDim2
// IDim0: this is iImgrows - quantity of rows of input power image, this is F
// IDim1: changes 
// IDim2: this is iImgcols - quantity of cols of input power image, this is T 
void fncFdmtIteration_cpu(int*  piarrInp, const float val_dF, const int IDim0, const int IDim1
	, const int IDim2, const int IMaxDT, const float VAlFmin
	, const float VAlFmax, const int ITerNum, float*  arr_val0
	, float*  arr_val1, int*  iarr_deltaTLocal
	, int*  iarr_dT_MI, int*  iarr_dT_ML, int*  iarr_dT_RI
	, int*  piarrOut, int& iOutPutDim0, int& iOutPutDim1)
{

	float valDeltaF = pow(2., ITerNum) * val_dF;
	float temp0 = 1. / (VAlFmin * VAlFmin) -
		1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));

	const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) - 1. / (VAlFmax * VAlFmax);

	int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
	iOutPutDim1 = ideltaT + 1;
	iOutPutDim0 = IDim0 / 2;


	// set zeros in output array
	memset( piarrOut, 0, iOutPutDim0 * iOutPutDim1 * IDim2 * sizeof(int));
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
	int F_Jumps = iOutPutDim0;
	
	

//#pragma omp parallel num_threads(nt) // OMP (главная директива для запуска нескольких потоков, количество потоков nt)
#pragma omp parallel // OMP (Если не указывать количество потоков nt, то по умолчанию будет использовано максимальное количество потоков)
	{ // OMP (начало блока, который выполняется в нескольких потоках
		for (int i_F = 0; i_F < F_Jumps; ++i_F)
		{

			float valf_start = VAlC2 * i_F + VAlFmin;
			float valf_end = VAlC2 * ( 1. + i_F) + VAlFmin;

			float valf_middle = (valf_end - valf_start)/2. + valf_start - val_correction;
			float valf_middle_larger = (valf_end - valf_start) / 2. + valf_start + val_correction;
			float temp0 = 1. / (valf_start * valf_start) - 1. / (valf_end * valf_end);

			arr_val0[i_F] = -(1. / (valf_middle * valf_middle) - 1. / (valf_start * valf_start)) / temp0;

			arr_val1[i_F] = -(1. / (valf_middle_larger * valf_middle_larger)
				- 1. / (valf_start * valf_start)) / temp0;

			iarr_deltaTLocal[i_F] = (int)(ceil((((float)(IMaxDT)) - 1.) * temp0 / VAlTemp1));
		
		}
	} // ! OMP (начало блока, который выполняется в нескольких потоках !
	
	

	// !10



	// 12. calculating second 3 auxillary 2 dim arrays
	

	create_2d_arrays_cpu(iOutPutDim0
		, iOutPutDim1,  arr_val0,  arr_val1,  iarr_deltaTLocal
		,  iarr_dT_MI,  iarr_dT_ML
		,  iarr_dT_RI);
	
	// !11

	// 13. 
	
	shift_and_sum_cpu( piarrInp
		, IDim0, IDim1, IDim2,  iarr_deltaTLocal,  iarr_dT_MI,  iarr_dT_ML,  iarr_dT_RI
		, iOutPutDim0, iOutPutDim1,  piarrOut);
	/*shift_and_sum_cpu_v1 ( piarrInp
		, IDim0, IDim1, IDim2,  iarr_deltaTLocal,  iarr_dT_MI,  iarr_dT_ML,  iarr_dT_RI
		, iOutPutDim0, iOutPutDim1,  piarrOut);*/


}

//-----------------------------------------------------------------------------------------------------------------------

void shift_and_sum_cpu_v1(int*  piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int*  iarr_deltaTLocal, int*  iarr_dT_MI
	, int*  iarr_dT_ML, int*  iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int*  piarrOut)
{
	int iw = IOutPutDim1 * IDim2;
#pragma omp parallel // OMP (Если не указывать количество потоков nt, то по умолчанию будет использовано максимальное количество потоков)
	{ // OMP (начало блока, который выполняется в нескольких потоках
		for (int i = 0; i < IOutPutDim0 * IOutPutDim1 * IDim2; ++i)
		{
			int i_F = i / iw;
			int irest = i % iw;
			int i_dT = irest / IDim2;
			if (i_dT > iarr_deltaTLocal[i_F])
			{
				continue;
			}
			int idx = irest % IDim2;
			// claculation of bound index: 
			// arr_dT_ML[i_F, i_dT]
			// index of arr_dT_ML
			// arr_dT_ML is matrix with IOutPutDim0 rows and IOutPutDim1 cols
			int ind = i_F * IOutPutDim1 + i_dT;
			// !

			// calculation of:
			//  Output[i_F][i_dT][idx] =  input[2 * i_F][arr_dT_MI[i_F, i_dT]][idx]
			  // calculation num row of submatix No_2 * i_F of  piarrInp = arr_dT_MI[ind]
			piarrOut[i] = piarrInp[2 * i_F * IDim1 * IDim2 +
				iarr_dT_MI[ind] * IDim2 + idx];

			if (idx >= iarr_dT_ML[ind])
			{
				int numRow = iarr_dT_RI[ind];
				int indInpMtrx = (2 * i_F + 1) * IDim1 * IDim2 + numRow * IDim2 + idx - iarr_dT_ML[ind];				
				piarrOut[i] += piarrInp[indInpMtrx];
			}
		}
	}
	
}

//-----------------------------------------------------------------------------------------------------------------------

void shift_and_sum_cpu(int*  piarrInp, const int IDim0, const int IDim1
	, const int IDim2, int*  iarr_deltaTLocal, int*  iarr_dT_MI
	, int*  iarr_dT_ML, int*  iarr_dT_RI, const int IOutPutDim0, const int IOutPutDim1
	, int*  piarrOut)
{
#pragma omp parallel // OMP (Если не указывать количество потоков nt, то по умолчанию будет использовано максимальное количество потоков)
	{ // OMP (начало блока, который выполняется в нескольких потоках	
		for (int i_F = 0; i_F < IOutPutDim0; ++i_F)
		{

			for (int i_dT = 0; i_dT < (1 + iarr_deltaTLocal[i_F]); ++i_dT)
			{
				int numRowOutputMtrxBegin0 = i_F * IOutPutDim1 * IDim2 + i_dT * IDim2;
				// number of element of beginning of the input 2 * i_F matrix's row with number 
				// dT_middle_index[i_F][i_dT]
				int numRowInputMtrxBegin0 = 2 * i_F * IDim1 * IDim2 + IDim2 * (iarr_dT_MI[i_F * IOutPutDim1 + i_dT]);
				memcpy(&piarrOut[numRowOutputMtrxBegin0], &piarrInp[numRowInputMtrxBegin0], IDim2 * sizeof(int));

				// number of beginning element of summated rows
				int numElemInRow = iarr_dT_ML[i_F * IOutPutDim1 + i_dT];
				// number of beginning element of output matrix  Output[i_F, i_dT, dT_middle_larger:]
				int numRowOutputMtrxBegin1 = numRowOutputMtrxBegin0 + numElemInRow;

				// number of the row of the submatrix of input matrix with number 2 * i_F + 1
				int numRowOfInputSubmatrix = iarr_dT_RI[i_F * IOutPutDim1 + i_dT];
				// number of beginning element of the input matrix Input[2 * i_F + 1, dT_rest_index, :i_T_max - dT_middle_larger]
				int numRowInputMtrxBegin1 = (2 * i_F + 1) * IDim1 * IDim2 + IDim2 * numRowOfInputSubmatrix;
				for (int j = 0; j < (IDim2 - numElemInRow); ++j)
				{
					piarrOut[numRowOutputMtrxBegin1 + j] += piarrInp[numRowInputMtrxBegin1 + j];
				}


			}
		}
	}
	
}

//--------------------------------------------------------------------------------------

void create_2d_arrays_cpu(const int IDim0, const int IDim1
	, float*  arr_val0, float*  arr_val1, int*  iarr_deltaTLocal
	, int*  iarr_dT_middle_index, int*  iarr_dT_middle_larger
	, int*  iarr_dT_rest_index)

{
    #pragma omp parallel // OMP (Если не указывать количество потоков nt, то по умолчанию будет использовано максимальное количество потоков)
	{ // OMP (начало блока, который выполняется в нескольких потоках

		for (int i_F = 0; i_F < IDim0; ++i_F)
		{
			if (i_F == 4)
			{
				int ii = 0;
			}
			for (int i_dT = 0; i_dT < IDim1; ++i_dT)
			{
				int i = i_F * IDim1 + i_dT;
				if (i_dT > (iarr_deltaTLocal[i_F]))
				{
					iarr_dT_middle_index[i] = 0;
					iarr_dT_middle_larger[i] = 0;
					iarr_dT_rest_index[i] = 0;
					continue;
				}
				iarr_dT_middle_index[i] = round(((float)i_dT) * arr_val0[i_F]);
				int ivalt = round(((float)i_dT) * arr_val1[i_F]);
				iarr_dT_middle_larger[i] = ivalt;
				iarr_dT_rest_index[i] = i_dT - ivalt;
			}
		}
	}	
}
//--------------------------------------------------------------------------------------
void fnc_init_cpu(int*  piarrImg, const int IImgrows, const int IImgcols
	, const int IDeltaT, int*  piarrOut)
{
	
	memset( piarrOut, 0, IImgrows * IImgcols * (IDeltaT + 1) * sizeof(int));
    #pragma omp parallel // OMP (Если не указывать количество потоков nt, то по умолчанию будет использовано максимальное количество потоков)
	{ // OMP (начало блока, который выполняется в нескольких потоках
		for (int i = 0; i < IImgrows; ++i)
		{
			{
				memcpy(&piarrOut[i * (IDeltaT + 1) * IImgcols], &piarrImg[i * IImgcols]
					, IImgcols * sizeof(int));
			}
		}
	}

    #pragma omp parallel // OMP (Если не указывать количество потоков nt, то по умолчанию будет использовано максимальное количество потоков)
	{ // OMP (начало блока, который выполняется в нескольких потоках
		for (int i_dT = 1; i_dT < (IDeltaT + 1); ++i_dT)
			for (int iF = 0; iF < IImgrows; ++iF)
			{
				int* result = &piarrOut[iF * (IDeltaT + 1) * IImgcols + i_dT * IImgcols + i_dT];
				int* arg0 = &piarrOut[iF * (IDeltaT + 1) * IImgcols + (i_dT - 1) * IImgcols + i_dT];
				int* arg1 = &piarrImg[iF * IImgcols];
				for (int j = 0; j < (IImgcols - i_dT); ++j)
				{
					result[j] = arg0[j] + arg1[j];
				}
			}
	}

}


////-------------------------------------------------------------------------------------------------------------------------------
//void fncCalcDimensionsOfOutputArrays(std::vector<int>* pivctOutDim0, std::vector<int>* pivctOutDim1
//	, std::vector<int>* pivctOutDim2, const int IDim0, const int IDim1
//	, const int IDim2, const int IMaxDT, const float VAlFmin
//	, const float VAlFmax)
//{
//	float val_dF = (VAlFmax - VAlFmin) / ((float)((*pivctOutDim0)[0]));
//
//
//
//	for (int it = 1; it < pivctOutDim0->size(); ++it)
//	{
//		float valDeltaF = pow(2., it) * val_dF;
//		float temp0 = 1. / (VAlFmin * VAlFmin) -
//			1. / ((VAlFmin + valDeltaF) * (VAlFmin + valDeltaF));
//		const float VAlTemp1 = 1. / (VAlFmin * VAlFmin) -
//			1. / (VAlFmax * VAlFmax);
//		int ideltaT = (int)(ceil(((float)IMaxDT - 1.0) * temp0 / VAlTemp1));
//		(*pivctOutDim1)[it] = ideltaT + 1;
//		(*pivctOutDim0)[it] = (*pivctOutDim0)[it - 1] / 2;
//		(*pivctOutDim2)[it] = (*pivctOutDim2)[it - 1];
//
//
//	}
//}

