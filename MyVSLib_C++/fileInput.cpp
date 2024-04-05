//THIS FILE CONTAINS FUNCTION TO READ FOUR <*>.NPY FILES WITH INPUT INFORMATION

//#define _CRT_SECURE_NO_WARNINGS
#include "fileInput.h"
#include <math.h>
#include <iostream>
#include "npy.hpp"

using namespace std;


int downloadInputData(char* strFolder, int* iMaxDT, int** ppiarrImage
	, int* iImRows, int* iImCols, float* val_fmin, float* val_fmax)
{
	bool fortran_order = false;
	// 1. loading typeofdata
	std::vector<unsigned long> shape{};
	std::vector<int> imaxDT;
	char arrch0[] = "//imaxDT.npy";
	char chpath0[100] = { 0 };
	strcpy(chpath0, strFolder);
	strcat(chpath0, arrch0);
	npy::LoadArrayFromNumpy_(chpath0, shape, fortran_order, imaxDT);
	//npy::LoadArrayFromNumpy(chpath0, shape,  imaxDT);
	*iMaxDT = imaxDT[0];
	// !1

	// 2. loading XX
	char arrch1[] = "//XX.npy";
	char chpath1[100] = { 0 };
	strcpy(chpath1, strFolder);
	strcat(chpath1, arrch1);
	std::vector<unsigned long> shape1{};
	std::vector<int> vctXX;
	npy::LoadArrayFromNumpy_(chpath1, shape1, fortran_order, vctXX);
	//npy::LoadArrayFromNumpy(chpath1, shape1,  vctXX);
	// !2

	// 3. loading shape
	char arrch2[] = "//iarrShape.npy";
	char chpath2[100] = { 0 };
	strcpy(chpath2, strFolder);
	strcat(chpath2, arrch2);
	std::vector<unsigned long> shape0{};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy_(chpath2, shape0, fortran_order, ivctImShape);
	//npy::LoadArrayFromNumpy(chpath2, shape0,  ivctImShape);
	*iImRows = ivctImShape[0];
	*iImCols = ivctImShape[1];
	// !3

	// 4. loading fmin and fmax 
	char arrch3[] = "//fmin_max.npy";
	char chpath3[100] = { 0 };
	strcpy(chpath3, strFolder);
	strcat(chpath3, arrch3);
	std::vector<unsigned long> shape2{};
	std::vector<float> vctfmin_max;
	npy::LoadArrayFromNumpy_(chpath3, shape2, fortran_order, vctfmin_max);
	//npy::LoadArrayFromNumpy(chpath3, shape2,  vctfmin_max);
	*val_fmin = vctfmin_max[0];
	*val_fmax = vctfmin_max[1];
	// ! 4

	// 5.checking dimensions
	bool bDim0_OK = false, bDim1_OK = false;
	int numcur = 2;
	for (int i = 1; i < 31; ++i)
	{
		numcur = 2 * numcur;
		if (numcur == ivctImShape[0])
		{
			bDim0_OK = true;
		}
		if (numcur == ivctImShape[1])
		{
			bDim1_OK = true;
		}
	}
	if (!(bDim0_OK && bDim1_OK)) {

		return 2;
	}
	// ! 5

	// 6. realloc and fill array Image

	size_t size = (size_t)(vctXX.size() * sizeof(int));

	if (!(*ppiarrImage = (int*)realloc(*ppiarrImage, size)))
	{
		return 1;
	}

	for (int i = 0; i < vctXX.size(); ++i)
	{
		(*ppiarrImage)[i] = vctXX[i];

	}
	// !6

	return 0;
}
//-------------------------------------------------------------------------------------
int downloadInputData_gpu(char* strFolder, int* iMaxDT, float* piarrImage
	, int* iImRows, int* iImCols, float* val_fmin, float* val_fmax)
{
	bool fortran_order = false;
	// 1. loading typeofdata
	std::vector<unsigned long> shape{};
	std::vector<int> imaxDT;
	char arrch0[] = "//imaxDT.npy";
	char chpath0[100] = { 0 };
	strcpy(chpath0, strFolder);
	strcat(chpath0, arrch0);
	npy::LoadArrayFromNumpy_(chpath0, shape, fortran_order, imaxDT);
	//npy::LoadArrayFromNumpy(chpath0, shape,  imaxDT);
	*iMaxDT = imaxDT[0];
	// !1

	// 2. loading XX
	char arrch1[] = "//XX.npy";
	char chpath1[100] = { 0 };
	strcpy(chpath1, strFolder);
	strcat(chpath1, arrch1);
	std::vector<unsigned long> shape1{};
	std::vector<int> vctXX;
	npy::LoadArrayFromNumpy_(chpath1, shape1, fortran_order, vctXX);
	//npy::LoadArrayFromNumpy(chpath1, shape1,  vctXX);
	// !2

	// 3. loading shape
	char arrch2[] = "//iarrShape.npy";
	char chpath2[100] = { 0 };
	strcpy(chpath2, strFolder);
	strcat(chpath2, arrch2);
	std::vector<unsigned long> shape0{};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy_(chpath2, shape0, fortran_order, ivctImShape);
	//npy::LoadArrayFromNumpy(chpath2, shape0,  ivctImShape);
	*iImRows = ivctImShape[0];
	*iImCols = ivctImShape[1];
	// !3

	// 4. loading fmin and fmax 
	char arrch3[] = "//fmin_max.npy";
	char chpath3[100] = { 0 };
	strcpy(chpath3, strFolder);
	strcat(chpath3, arrch3);
	std::vector<unsigned long> shape2{};
	std::vector<float> vctfmin_max;
	npy::LoadArrayFromNumpy_(chpath3, shape2, fortran_order, vctfmin_max);
	//npy::LoadArrayFromNumpy(chpath3, shape2,  vctfmin_max);
	*val_fmin = vctfmin_max[0];
	*val_fmax = vctfmin_max[1];
	// ! 4

	// 5.checking dimensions
	bool bDim0_OK = false, bDim1_OK = false;
	int numcur = 2;
	for (int i = 1; i < 31; ++i)
	{
		numcur = 2 * numcur;
		if (numcur == ivctImShape[0])
		{
			bDim0_OK = true;
		}
		if (numcur == ivctImShape[1])
		{
			bDim1_OK = true;
		}
	}
	if (!(bDim0_OK && bDim1_OK)) {

		return 2;
	}
	// ! 5

	// 6. fill array Image

	size_t size = (size_t)(vctXX.size() * sizeof(int));


	for (int i = 0; i < vctXX.size(); ++i)
	{
		piarrImage[i] = (float)vctXX[i];

	}
	// !6

	return 0;
}

//----------------------------------------------------
int readHeader(char* chInpFilePass, unsigned int& lenarr, unsigned int& n_p
	, float& valD_max_, float& valf_min_, float& valf_max_, float& valSigmaBound_)
{
	FILE* file = fopen(chInpFilePass, "rb");
	if (file == nullptr) {
		std::cerr << "Error opening file." << std::endl;
		return 1;
	}


	// Read the integer variables
	fread(&lenarr, sizeof(int), 1, file);
	fread(&n_p, sizeof(int), 1, file);

	// Read the float variables
	fread(&valD_max_, sizeof(float), 1, file);
	fread(&valf_min_, sizeof(float), 1, file);
	fread(&valf_max_, sizeof(float), 1, file);
	fread(&valSigmaBound_, sizeof(float), 1, file);
	fclose(file);
}
int readDimensions(char* strFolder, int* iImRows, int* iImCols)
{	
	bool fortran_order = false;
	char arrch2[] = "//iarrShape.npy";
	char chpath2[100] = { 0 };
	strcpy(chpath2, strFolder);
	strcat(chpath2, arrch2);
	std::vector<unsigned long> shape0{};
	std::vector<int> ivctImShape;
	npy::LoadArrayFromNumpy_(chpath2, shape0, fortran_order, ivctImShape);
	
	*iImRows = ivctImShape[0];
	*iImCols = ivctImShape[1];
	return 0;
	
}