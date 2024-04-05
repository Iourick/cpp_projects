
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "FdmtGpu.cuh"
#include <vector>
#include "npy.hpp"
#include "kernel.cuh"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"

using namespace std;

char strInpFolder[] = "..//FDMT_TESTS//512";
char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";
const bool BDIM_512_1024 = true;


void printDeviceInfo()
{
	int deviceId;
	int numberOfSMs;
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties(&deviceProps, deviceId);
	std::string deviceName = deviceProps.name;
	std::cout << "Device Name: " << deviceName << std::endl;
	std::cout << "Number of SM: " << numberOfSMs << std::endl;
}


//---------------------------------------

int main(int argc, char** argv)
{
	printDeviceInfo();
	//--------------------------------------------------------------------------------------------------------------
	//------------------- prepare to work -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------

	// initiating input variables
	int iMaxDT = 0;
	int iImRows = 0, iImCols = 0;
	float val_fmax = 0., val_fmin = 0.;
	readDimensions(strInpFolder, &iImRows, &iImCols);
	// initiate pointer to input image
	fdmt_type_* u_parrImage = 0;// (int*)malloc(sizeof(int));
	fdmt_type_* h_parrImage =  (fdmt_type_*)malloc(sizeof(fdmt_type_) * iImRows * iImCols);
	cudaMalloc/*Managed*/(&u_parrImage, sizeof(fdmt_type_) * iImRows * iImCols);

	// reading input files from folder 
	int ireturn = downloadInputData_gpu(strInpFolder, &iMaxDT, h_parrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax); 
	cudaMemcpy(u_parrImage, h_parrImage, sizeof(fdmt_type_) * iImRows * iImCols, cudaMemcpyHostToDevice);
	free(h_parrImage);
	iMaxDT = iMaxDT ;//iMaxDT * 3 +10;//iMaxDT//;
	int  nchan =iImRows;// 400;

	// analysis output of reading function
	switch (ireturn)
	{
	case 1:
		cout << "Err. ...Can't allocate memory for input image. Oooops... " << std::endl;
		return 1;
	case 2:
		cout << "Err. ...Input dimensions must be a power of 2. Oooops... " << std::endl;
		return 1;
	case 0:
		cout << "Input data downloaded properly " << std::endl;
		break;
	default:
		cout << "..something extraordinary happened! Oooops..." << std::endl;
		break;
	}	
	// !	

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of prepare ------------------------------------------------------------------------------------------
	//------------------- begin to calculate cuda var -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	//iImCols = 1 << 18;
	CFdmtGpu* pfdmt = new CFdmtGpu(
		 val_fmin
		, val_fmin +  (val_fmax - val_fmin)/ iImRows * nchan
		, nchan
		, iImCols
		, iMaxDT
	);	
	
	

	// 1. allocate memory for output:
	fdmt_type_* u_parrImOut = 0;
	cudaMallocManaged(&u_parrImOut, iImCols * iMaxDT * sizeof(fdmt_type_));
	
	
	// 2!
	
	// 3. calculations		
	int num = 100;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num; ++i)
	{
		pfdmt->process_image(u_parrImage       // on-device input image			
			, u_parrImOut	// OUTPUT image, dim = IMaxDT x IImgcols
			, false);
		
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "Time taken by function fncFdmtU_cu: " << duration.count() / ((double)num) << " microseconds" << std::endl;

	delete pfdmt;
	// !3
	
	//4. write  output in .npy:IImgcols * IMaxDT * sizeof(int));
	cudaDeviceSynchronize();
	fdmt_type_* parrImOut = (fdmt_type_*)malloc(iImCols * iMaxDT * sizeof(fdmt_type_));
	cudaMemcpy(parrImOut, u_parrImOut, iImCols * iMaxDT * sizeof(fdmt_type_), cudaMemcpyDeviceToHost);

	//std::vector<float> v1(parrImOut, parrImOut + iImCols * iMaxDT);
	std::vector<float> v1(iImCols * iMaxDT);
	for (int i = 0; i < iImCols * iMaxDT; ++i)
	{
		v1.at(i) = (float)parrImOut[i];
	}

	std::array<long unsigned, 2> leshape101{ iImCols ,iMaxDT};

	npy::SaveArrayAsNumpy(strPathOutImageNpyFile_gpu, false, leshape101.size(), leshape101.data(), v1);

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of calculations ------------------------------------------------------------------------------------------
	//------------------- begin to draw output image for cuda -------------------------------------------------------------------------------------------

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

	cout << "GFLP/sec = " << ((double)flops) / ((double)duration.count() / ((double)num)) * 1.0e6 << endl;

	
	free(parrImOut);		
	cudaFree(u_parrImage);
	cudaFree(u_parrImOut);

	char filename_cpu[] = "image_gpu.png";
	createImg_(argc, argv, v1, iMaxDT, iImCols, filename_cpu);

	return 0;
}
