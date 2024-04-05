// fdmt_cpu.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <iostream>

#include <math.h>
#include <stdio.h>
#include <iostream>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <stdlib.h>
#include "fileInput.h"
#include "FdmtCpu.h"
#include "Constants.h"


#define DRAW true 
#if DRAW == true
#include "DrawImg.h"
#endif
using namespace std;
char strInpFolder[] = "..//FDMT_TESTS//512";
char strPathOutImageNpyFile[] = "out_image_CPU.npy";
const bool BDIM_512_1024 = false;

extern int quantFlops = 0;

int main(int argc, char** argv)
{
	
	//--------------------------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	//------------------- prepare to work -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	// initiate pointer to input image
	int* piarr = (int*)malloc(sizeof(int));

	// initiate 2-pointer to input image, in order to realloc memory to satisfy arbitrary dimensions
	int** ppiarrImage = &piarr;

	// initiating input variables
	int iMaxDT = 0;
	int iImRows = 0, iImCols = 0;
	float val_fmax = 0., val_fmin = 0.;

	// reading input files from folder 
	int ireturn = downloadInputData(strInpFolder, &iMaxDT, ppiarrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax);

	// output's analys   of reading function
	switch (ireturn)
	{
	case 1:
		cout << "Err. Can't allocate memory for input image. Oooops... " << std::endl;
		return 1;
	case 2:
		cout << "Err. Input dimensions must be a power of 2. Oooops... " << std::endl;
		return 1;
	case 0:
		cout << "Input data downloaded properly. Congradulations! " << std::endl;

		break;
	default:
		cout << "Happened something extraordinary! Oooops..." << std::endl;
		break;
	}

	// 5.1
	if ((iImRows == 1024) && (BDIM_512_1024))
	{
		iImRows = 512;
		iMaxDT = 512;
		piarr = (int*)realloc(piarr, iImRows * iImCols * sizeof(int));

	}
	// ! 5.1

	// declare constants
	const int IMaxDT =  iMaxDT;
	const int IImgrows = 259;// iImRows;
	const int IImgcols =  iImCols;
	const float VAlFmin = val_fmin;
	const float VAlFmax = val_fmax;


	// handy pointer to input image
	int* piarrImage = *ppiarrImage;

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of prepare ------------------------------------------------------------------------------------------
	//------------------- begin to work -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	// 1. allocate memory for device array

	int* piarrImOut = (int*)malloc(IImgcols * IMaxDT * sizeof(int));
	if (NULL == piarrImOut)
	{
		return 1;
	}
	// !1

	// 2. calculations	
	clock_t start = clock()*1000.;
	
	CFdmtCpu* fdmt = new CFdmtCpu(
		VAlFmin
		, VAlFmin + IImgrows * (VAlFmax - VAlFmin) / iImRows
		, IImgrows// quant channels/rows of input image
		, IImgcols
		, IMaxDT// quantity of rows of output image
	);
	
	fdmt->process_image(piarrImage, piarrImOut, false);
	
	clock_t end = clock() * 1000.;
	double duration = double(end - start) / CLOCKS_PER_SEC;
	std::cout << "Succeeded. Time taken by function fncFdmt_cpu_v0: " << duration << " miliseconds" << std::endl;

	// !2

	// write output image in <*>.NPY file with path strPathOutImageNpyFile:

	std::vector<int> v1(piarrImOut, piarrImOut + IImgcols * IMaxDT);

	std::array<long unsigned, 2> leshape101 { IMaxDT,IImgcols};

	npy::SaveArrayAsNumpy(strPathOutImageNpyFile, false, leshape101.size(), leshape101.data(), v1);
	free(piarr);
	
	float flops = 0;
	if (iImRows == 512)
	{
		flops = GFLPS_512;
	}
	else
	{
		if (iImRows == 1024)
		{
			if (BDIM_512_1024)
			{
				flops = GFLPS_512_1024;
			}
			else
			{
				flops = GFLPS_1024;
			}
		}
		else
		{
			flops = GFLPS_2048;
		}
	}
	cout << "GFLP = " << flops << "  GFP" << endl;
	cout << "GFLP/sec = " << ((double)flops) / ((double)duration) * 1000. << "  GFP" << endl;	
	cout << "FLOPS = " << quantFlops << endl;
	
	free(piarrImOut);
	delete fdmt;
 #if DRAW == true
	char filename_cpu[] = "image_cpu.png";
	
	createImg_(argc, argv, v1, IMaxDT, IImgcols,  filename_cpu);
#endif

	return 0;
}


