
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "npy.hpp"
#include "kernel.cuh"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"
#include "fdmt.h"

using namespace std;
enum TYPE_OF_PROCESSOR
{
	CPU
	, GPU
};

char strInpFolder[] = "..//FDMT_TESTS//2048";
char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";
const bool BDIM_512_1024 = true;
TYPE_OF_PROCESSOR PROCESSOR = CPU;


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
	//int  nchan = iImRows;// 400;
	readDimensions(strInpFolder, &iImRows, &iImCols);
	// initiate pointer to input image
	fdmt_type_* h_parrImage = (fdmt_type_*)malloc(sizeof(fdmt_type_) * iImRows * iImCols);
	int ireturn = downloadInputData_gpu(strInpFolder, &iMaxDT, h_parrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax);

	fdmt_type_* u_parrImage = NULL;
	fdmt_type_* u_parrImOut = NULL;
	float tsamp = 1.0;
	
	size_t dt_step = 8;
	size_t dt_min = 1;
	FDMT* pfdmt = new FDMT(val_fmin, val_fmax, iImRows, iImCols, tsamp,
		iMaxDT, dt_step, dt_min);
	const size_t IOutImageRows = pfdmt->get_dt_grid_final().size();
	size_t dmt_size = IOutImageRows * iImCols;
	u_parrImOut = (fdmt_type_*)malloc(dmt_size * sizeof(fdmt_type_));
	if (PROCESSOR == GPU)
	{
		cudaMalloc(&u_parrImage, sizeof(fdmt_type_) * iImRows * iImCols);
		cudaMemcpy(u_parrImage, h_parrImage, sizeof(fdmt_type_) * iImRows * iImCols, cudaMemcpyHostToDevice);
		cudaMallocManaged(&u_parrImOut, iImCols * iMaxDT * sizeof(fdmt_type_));
		
	}
	else
	{
		u_parrImage = (fdmt_type_*)malloc(sizeof(fdmt_type_) * iImRows * iImCols);
		memcpy(u_parrImage, h_parrImage, sizeof(fdmt_type_) * iImRows * iImCols);
		u_parrImOut = (fdmt_type_*)malloc(sizeof(fdmt_type_) * iMaxDT * iImCols);
		
	}
	free(h_parrImage);
	
	iMaxDT = iMaxDT ;//iMaxDT * 3 +10;//iMaxDT//;	

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of prepare ------------------------------------------------------------------------------------------
	//------------------- begin to calculate cuda var -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	//iImCols = 1 << 18;
	
	
	// 3. calculations		
	int num = 100;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num; ++i)
	{
		pfdmt->execute(u_parrImage       // on-device input image	
			, iImRows * iImCols
			, u_parrImOut	// OUTPUT image, dim = IMaxDT x IImgcols
			, dmt_size);
		
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "PROCESSOR = "<< PROCESSOR<<"  Time taken by function fncFdmtU_cu : " << duration.count() / ((double)num) << " microseconds" << std::endl;

	delete pfdmt;
	// !3
	
	//4. write  output in .npy:IImgcols * IMaxDT * sizeof(int));
	//cudaDeviceSynchronize();
	fdmt_type_* parrImOut = (fdmt_type_*)malloc(iImCols * IOutImageRows * sizeof(fdmt_type_));

	if (PROCESSOR == GPU)
	{		
		cudaMemcpy(parrImOut, u_parrImOut, iImCols * IOutImageRows * sizeof(fdmt_type_), cudaMemcpyDeviceToHost);
	}
	else
	{
		memcpy(parrImOut, u_parrImOut, iImCols * IOutImageRows * sizeof(fdmt_type_));
	}
	

	//std::vector<float> v1(parrImOut, parrImOut + iImCols * iMaxDT);
	std::vector<float> v1(iImCols * IOutImageRows);
	for (int i = 0; i < iImCols * IOutImageRows; ++i)
	{
		v1.at(i) = (float)parrImOut[i];
	}

	std::array<long unsigned, 2> leshape101{ iImCols ,IOutImageRows };

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

	if (PROCESSOR == GPU)
	{
		cudaFree(u_parrImage);
		cudaFree(u_parrImOut);
	}
	else
	{
		free(u_parrImage);
		free(u_parrImOut);
	}
	free(parrImOut);		
	

	char filename_cpu[] = "image_gpu.png";
	createImg_(argc, argv, v1, IOutImageRows, iImCols, filename_cpu);

	return 0;
}
