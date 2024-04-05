#include "Constants.h"
//The module contains signal detection functions detect_signal_gpu(..).
//Signal detection is performed in the output image obtained 
//from the algorithm FDMT. Detection is carried out by comparing the SNR 
//of the accumulated signal with a threshold value.
//INPUT:
//d_arr - output matrix from FDMT algirithm, allocated on GPU
//Rows, Cols - quantity of rows and columns of the d_arr respectively
//d_norm - matrix with apriori noise values, allocated on GPU, with the 
//same dimentions as matrix d_arr
//WndWidth - maximal length of summation window, enumeration strats from value =1.
//gridSize, blockSize - needed variables to launch kernel
//should have the following relation with Rows and Cols:
//    const dim3 blockSize = dim3(512, 1, 1);
//    const dim3 gridSize = dim3((Cols + blockSize.x - 1) / blockSize.x, Rows, 1);
//    It is possible to use any relevant number instead of 512 
//d_pAuxArray, d_pAuxNumArray, d_pWidthArray - auxillary arrays with length gridSize.x* gridSize.y
//pstructOut - the structure contains  the best candidate's parameters (SNR, number of row and column,
//length of summation window)
//function returns TRUE if detection is sucessful and FALSE if not.
//Detection is being done with accordance with the following algorithm:
//    fix up number of checking element of d_arr i,j
//        calculate
//        sig2 = sum (d_arr[i][j + k]* d_arr[i][j + k]),
//        noise2 = sum(norm[i][j + k] * norm[i][j + k])
//        k=0,..,iw-1
//        calculate:
//        SNR = SQRT(sig2/ noise2)
//        compare with treshold:
//    if(SNR >= valTresh) -> pare (i,j) - is candidate


struct structOutDetection
{
    int iwidth;
    int irow;
    int icol;
    float snr;
};

void detect_signal_gpu(fdmt_type_* d_arr, fdmt_type_* d_norm, const int Rows
    , const int  Cols, const int WndWidth, const dim3 gridSize, const dim3 blockSize
    , float* d_pAuxArray, int* d_pAuxNumArray, int* d_pWidthArray, structOutDetection* pstructOut);

__global__
void multi_windowing_kernel(fdmt_type_* d_arr, fdmt_type_* d_norm, const int  Cols
    , const int WndWidth, float* d_pAuxArray, int* d_pAuxIntArray, int* d_pWidthArray);

__global__
void complete_detection_kernel(const int Cols, const int LEnGrid, float* d_pAuxArray, int* d_pAuxNumArray
    , int* d_pWidthArray, structOutDetection* pstructOut);

void multi_windowing_cpu(fdmt_type_* arr, fdmt_type_* norm, const int  Cols
    , const int WndWidth, float* pAuxArray, int* pAuxIntArray, int* pWidthArray, const dim3 gridSize, const dim3 blockSize);

__global__
void calcWindowedImage_kernel(fdmt_type_* d_arr, fdmt_type_* d_norm, const int  Cols
    , const int WndWidth, float* d_pOutArray);