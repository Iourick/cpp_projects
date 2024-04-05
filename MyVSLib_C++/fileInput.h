#pragma once
#include "Constants.h"
int downloadInputData(char* strFolder, int* iMaxDT, int** ppiarrImage
	, int* iImRows, int* iImCols, float* val_fmin, float* val_fmax);

int readHeader(char* chInpFilePass, unsigned int& lenarr, unsigned int& n_p
	, float& valD_max_, float& valf_min_, float& valf_max_, float& valSigmaBound_);

int readDimensions(char* strFolder, int* iImRows, int* iImCols);

int downloadInputData_gpu(char* strFolder, int* iMaxDT, float* piarrImage
	, int* iImRows, int* iImCols, float* val_fmin, float* val_fmax);