
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include "npy.hpp"
#include "kernel.cuh"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"
#include "fdmt_cpu.hpp"
#include  "fdmt_base.hpp"
#include "fdmt_gpu.cuh"
#include <span>
#include <fdmt_v1_gpu.cuh>




using namespace std;
enum TYPE_OF_PROCESSOR
{
	CPU
	, GPU
};

char strInpFolder[] = "..//FDMT_TESTS//2048";
char strPathOutImageNpyFile_gpu[] = "out_image_GPU.npy";
const bool BDIM_512_1024 = true;
TYPE_OF_PROCESSOR PROCESSOR = GPU;



#ifdef _WIN32
#include <comdef.h>
#include <Wbemidl.h>
#pragma comment(lib, "wbemuuid.lib")

void GetCPUInfo()
{
	HRESULT hres;

	// Initialize COM.
	hres = CoInitializeEx(0, COINIT_MULTITHREADED);
	if (FAILED(hres))
	{
		std::cerr << "Failed to initialize COM library. Error code = 0x"
			<< std::hex << hres << std::endl;
		return;
	}

	// Set general COM security levels.
	hres = CoInitializeSecurity(
		NULL,
		-1,                          // COM authentication
		NULL,                        // Authentication services
		NULL,                        // Reserved
		RPC_C_AUTHN_LEVEL_DEFAULT,   // Default authentication 
		RPC_C_IMP_LEVEL_IMPERSONATE, // Default Impersonation  
		NULL,                        // Authentication info
		EOAC_NONE,                   // Additional capabilities 
		NULL                         // Reserved
	);

	if (FAILED(hres))
	{
		std::cerr << "Failed to initialize security. Error code = 0x"
			<< std::hex << hres << std::endl;
		CoUninitialize();
		return;
	}

	// Obtain the initial locator to WMI.
	IWbemLocator* pLoc = NULL;

	hres = CoCreateInstance(
		CLSID_WbemLocator,
		0,
		CLSCTX_INPROC_SERVER,
		IID_IWbemLocator, (LPVOID*)&pLoc);

	if (FAILED(hres))
	{
		std::cerr << "Failed to create IWbemLocator object. Error code = 0x"
			<< std::hex << hres << std::endl;
		CoUninitialize();
		return;
	}

	// Connect to WMI through the IWbemLocator::ConnectServer method.
	IWbemServices* pSvc = NULL;

	hres = pLoc->ConnectServer(
		_bstr_t(L"ROOT\\CIMV2"), // Object path of WMI namespace
		NULL,                    // User name. NULL = current user
		NULL,                    // User password. NULL = current
		0,                       // Locale. NULL indicates current
		NULL,                    // Security flags.
		0,                       // Authority (for example, Kerberos)
		0,                       // Context object 
		&pSvc                    // Pointer to IWbemServices proxy
	);

	if (FAILED(hres))
	{
		std::cerr << "Could not connect. Error code = 0x"
			<< std::hex << hres << std::endl;
		pLoc->Release();
		CoUninitialize();
		return;
	}

	// Set security levels on the proxy.
	hres = CoSetProxyBlanket(
		pSvc,                        // Indicates the proxy to set
		RPC_C_AUTHN_WINNT,           // RPC_C_AUTHN_xxx
		RPC_C_AUTHZ_NONE,            // RPC_C_AUTHZ_xxx
		NULL,                        // Server principal name 
		RPC_C_AUTHN_LEVEL_CALL,      // RPC_C_AUTHN_LEVEL_xxx 
		RPC_C_IMP_LEVEL_IMPERSONATE, // RPC_C_IMP_LEVEL_xxx
		NULL,                        // Client identity
		EOAC_NONE                    // Proxy capabilities 
	);

	if (FAILED(hres))
	{
		std::cerr << "Could not set proxy blanket. Error code = 0x"
			<< std::hex << hres << std::endl;
		pSvc->Release();
		pLoc->Release();
		CoUninitialize();
		return;
	}

	// Use the IWbemServices pointer to make requests of WMI.
	IEnumWbemClassObject* pEnumerator = NULL;
	hres = pSvc->ExecQuery(
		bstr_t("WQL"),
		bstr_t("SELECT * FROM Win32_Processor"),
		WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
		NULL,
		&pEnumerator);

	if (FAILED(hres))
	{
		std::cerr << "Query for operating system name failed. "
			<< "Error code = 0x"
			<< std::hex << hres << std::endl;
		pSvc->Release();
		pLoc->Release();
		CoUninitialize();
		return;
	}

	// Get the data from the query.
	IWbemClassObject* pclsObj = NULL;
	ULONG uReturn = 0;

	while (pEnumerator)
	{
		HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1,
			&pclsObj, &uReturn);

		if (0 == uReturn)
		{
			break;
		}

		VARIANT vtProp;

		// Get the value of the Name property.
		hr = pclsObj->Get(L"Name", 0, &vtProp, 0, 0);
		std::wcout << " CPU Name : " << vtProp.bstrVal << std::endl;
		VariantClear(&vtProp);

		// Get other properties similarly.
		// Example: hr = pclsObj->Get(L"MaxClockSpeed", 0, &vtProp, 0, 0);
		// std::wcout << " Max Clock Speed : " << vtProp.uintVal << " MHz" << std::endl;

		pclsObj->Release();
	}

	// Cleanup
	pSvc->Release();
	pLoc->Release();
	pEnumerator->Release();
	CoUninitialize();
}
#elif __linux__
#include <fstream>
#include <string>

void GetCPUInfo()
{
	std::ifstream cpuinfo("/proc/cpuinfo");
	std::string line;

	while (std::getline(cpuinfo, line))
	{
		if (line.find("model name") != std::string::npos)
		{
			std::cout << line << std::endl;
		}
	}
}
#endif


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

__global__ void copy_array_no_restrict(int* dst, const int* src, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		dst[idx] = src[idx];
	}
}

// CUDA kernel with restrict
__global__ void copy_array_restrict(int* __restrict dst, const int* __restrict src, size_t n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		dst[idx] = src[idx];
	}
}
//---------------------------------------

int main(int argc, char** argv)
{
	// test restrict
	const size_t n = 100000000; // Size of the arrays
	size_t size = n * sizeof(int);

	// Host arrays
	int* h_src = (int*)malloc(size);
	int* h_dst = (int*)malloc(size);

	// Initialize host array
	for (size_t i = 0; i < n; ++i) {
		h_src[i] = 1;
	}

	// Device arrays
	int* d_src, * d_dst;
	cudaMalloc((void**)&d_src, size);
	cudaMalloc((void**)&d_dst, size);

	// Copy data from host to device
	cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

	// Define block size and grid size
	int blockSize = 256;
	int gridSize = (n + blockSize - 1) / blockSize;

	// Timing variables
	cudaEvent_t start0, stop0;
	float time_no_restrict, time_restrict;
	cudaEventCreate(&start0);
	cudaEventCreate(&stop0);

	// Measure time for kernel without restrict
	cudaEventRecord(start0);
	copy_array_no_restrict << <gridSize, blockSize >> > (d_dst, d_src, n);
	cudaEventRecord(stop0);
	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&time_no_restrict, start0, stop0);

	// Measure time for kernel with restrict
	cudaEventRecord(start0);
	copy_array_restrict << <gridSize, blockSize >> > (d_dst, d_src, n);
	cudaEventRecord(stop0);
	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&time_restrict, start0, stop0);

	// Copy result back to host
	cudaMemcpy(h_dst, d_dst, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_src);
	cudaFree(d_dst);

	// Free host memory
	free(h_src);
	free(h_dst);

	// Print the elapsed times
	std::cout << "Time without restrict: " << time_no_restrict << " ms\n";
	std::cout << "Time with restrict: " << time_restrict << " ms\n";

	// Clean up timing events
	cudaEventDestroy(start0);
	cudaEventDestroy(stop0);

	// !test


	GetCPUInfo();

	printDeviceInfo();
	//--------------------------------------------------------------------------------------------------------------
	//------------------- prepare to work -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------

	// initiating input variables
	int iMaxDT = 0;
	int iImRows = 0, iImCols = 0;
	int iImRows1 = 0, iImCols1 = 0;
	float val_fmax = 0., val_fmin = 0.;
	//int  nchan = iImRows;// 400;
	readDimensions(strInpFolder, &iImRows, &iImCols);
	// initiate pointer to input image
	iImRows = 4096;
	iImCols = 1 << 16;
	
	fdmt_type_* h_parrImage = (fdmt_type_*)malloc(sizeof(fdmt_type_) * iImRows * iImCols);
	memset(h_parrImage, 0, sizeof(fdmt_type_) * iImRows * iImCols);
	int ireturn = downloadInputData_gpu(strInpFolder, &iMaxDT, h_parrImage, &iImRows, &iImCols,
		&val_fmin, &val_fmax);
	iImRows = 4096;
	iImCols = 1 << 16;
	iMaxDT = iImRows/2;
	fdmt_type_* u_parrImage = NULL;
	fdmt_type_* u_parrImOut = NULL;
	float tsamp = 1.0;
	
	size_t dt_step = 1;
	size_t dt_min = 0;

	FDMT* pfdmt_cpu = new FDMTCPU(val_fmin, val_fmax, iImRows, iImCols, tsamp,
		iMaxDT - 1, dt_step, dt_min);
	const size_t IOutImageRows = pfdmt_cpu->get_dt_grid_final().size();
	delete pfdmt_cpu;
	char str[50] = { 0 };
	if (PROCESSOR == GPU)
	{
		strcpy(str, "GPU");
	}
	else
	{
		strcpy(str, "CPU");
	}
	std::cout << "   TEST's DATA" << std::endl;
	std::cout << "processor:  " <<str<< std::endl;
	std::cout << "nsub = " << iImRows << "  nsamp = " << iImCols << std::endl;
	std::cout << "dt_max =  " << iMaxDT << " dt_min =  " << dt_min  << std::endl;
	std::cout<< "dt_step =  " << dt_step << std::endl;

	FDMT* pfdmt = nullptr;	
	
	size_t dmt_size = IOutImageRows * iImCols;
	u_parrImOut = (fdmt_type_*)malloc(dmt_size * sizeof(fdmt_type_));
	if (PROCESSOR == GPU)
	{
		cudaMalloc(&u_parrImage, sizeof(fdmt_type_) * iImRows * iImCols);
		cudaMemcpy(u_parrImage, h_parrImage, sizeof(fdmt_type_) * iImRows * iImCols, cudaMemcpyHostToDevice);
		cudaMallocManaged(&u_parrImOut, iImCols * IOutImageRows * sizeof(fdmt_type_));
		//pfdmt = new FDMT_v1_GPU(val_fmin, val_fmax, iImRows, iImCols, tsamp,
			pfdmt = new FDMTGPU(val_fmin, val_fmax, iImRows, iImCols, tsamp,
			iMaxDT - 1, dt_step, dt_min);		
	}
	else
	{
		u_parrImage = (fdmt_type_*)malloc(sizeof(fdmt_type_) * iImRows * iImCols);
		memcpy(u_parrImage, h_parrImage, sizeof(fdmt_type_) * iImRows * iImCols);
		u_parrImOut = (fdmt_type_*)malloc(sizeof(fdmt_type_) * IOutImageRows * iImCols);
		pfdmt = new FDMTCPU(val_fmin, val_fmax, iImRows, iImCols, tsamp,
			iMaxDT - 1, dt_step, dt_min);		
	}
	free(h_parrImage);
	
	

	//--------------------------------------------------------------------------------------------------------------
	//-------------------- end of prepare ------------------------------------------------------------------------------------------
	//------------------- begin to calculate cuda var -------------------------------------------------------------------------------------------
	//--------------------------------------------------------------------------------------------------------------
	//iImCols = 1 << 18;
	
	std::cout << "timing begin" << std::endl;
	// 3. calculations		
	int num = 150;
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < num; ++i)
	{
		pfdmt->execute(u_parrImage       // on-device input image
			, iImRows * iImCols
			, u_parrImOut	// OUTPUT image, dim = IMaxDT x IImgcols
			, dmt_size
			);		
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
