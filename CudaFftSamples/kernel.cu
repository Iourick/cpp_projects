
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include <complex>
#include <iostream>
#include <vector>
#include <cufft.h>

//#include "cufft_utils.h"
 // CUDA API error checking
//#ifndef CUDA_RT_CALL
//#define CUDA_RT_CALL( call )                                                                                           \
//    {                                                                                                                  \
//        auto status = static_cast<cudaError_t>( call );                                                                \
//        if ( status != cudaSuccess )                                                                                   \
//            fprintf( stderr,                                                                                           \
//                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
//                     "with "                                                                                           \
//                     "%s (%d).\n",                                                                                     \
//                     #call,                                                                                            \
//                     __LINE__,                                                                                         \
//                     __FILE__,                                                                                         \
//                     cudaGetErrorString( status ),                                                                     \
//                     status );                                                                                         \
//    }
//#endif  // CUDA_RT_CALL
//
//// cufft API error chekcing
//#ifndef CUFFT_CALL
//#define CUFFT_CALL( call )                                                                                             \
//    {                                                                                                                  \
//        auto status = static_cast<cufftResult>( call );                                                                \
//        if ( status != CUFFT_SUCCESS )                                                                                 \
//            fprintf( stderr,                                                                                           \
//                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
//                     "with "                                                                                           \
//                     "code (%d).\n",                                                                                   \
//                     #call,                                                                                            \
//                     __LINE__,                                                                                         \
//                     __FILE__,                                                                                         \
//                     status );                                                                                         \
//    }
//#endif  // CUFFT_CALL

__global__
void scaling_kernel(cufftComplex* data, int element_count, float scale) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    for (auto i = tid; i < element_count; i += stride) {
        data[tid].x *= scale;
        data[tid].y *= scale;
    }
}


int main(int argc, char* argv[])
{
    /*char arrch[16] = { 1,2,3,4
    ,5,6,7,8
    ,9,10,11,12
    ,13,14,15,16 };
    cufftComplex* d_arrch = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_arrch), sizeof(char) * 16);
    cudaMemcpyAsync(d_arrch, arrch, sizeof(char) * 16,  cudaMemcpyHostToDevice);

    cufftComplex* pcmparrRawSignalCur = nullptr;
    cudaMallocManaged(reinterpret_cast<void**>(&pcmparrRawSignalCur), sizeof(cufftComplex) * 16);


    cudaMemcpy(pcmparrRawSignalCur, reinterpret_cast <cufftComplex*> (d_arrch), 8 * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < 8; ++i)
    {
        std::printf("%f + %fj\n", pcmparrRawSignalCur[i].x, pcmparrRawSignalCur[i].y);
    }*/


    cufftResult res;
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int fft_size = 8;
    int batch_size = 1;// 2;
    int element_count = batch_size * fft_size;

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(element_count, 0);

    for (int i = 0; i < element_count; i++) {
        data[i] = data_type(i, -i);
    }

    std::printf("Input array:\n");
    for (auto& i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    cufftComplex* d_data = nullptr;
    res = cufftCreate(&plan);
    if (res != CUFFT_SUCCESS)
    {
        printf("ERROR: cufftCreate failed\n");
    }
                                                               
    std::printf("fft_size = %i\n", fft_size);
    std::printf("batch_size = %i\n", batch_size);
    res = cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size);

    if (res != CUFFT_SUCCESS)
    {
        printf("ERROR: cufftPlan1d failed\n");
    }
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    

    // Create device data arrays
    cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(data_type) * data.size());
    cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
        cudaMemcpyHostToDevice, stream);

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    res = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if (res != CUFFT_SUCCESS)
    {
        printf("ERROR: cufftPlan1d failed\n");
    }
    cufftSetStream(plan, stream);
    std::printf("QU-QU !! \n");

    cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::printf("Output array after Forward FFT :\n");
    for (auto& i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");
    // Normalize the data
    scaling_kernel << <1, 128, 0, stream >> > (d_data, element_count, 1.f / fft_size);

    cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::printf("Output array after Forward FFT, Normalization :\n");
    for (auto& i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    // The original data should be recovered after Forward FFT, normalization and inverse FFT
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    

    cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
        cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    std::printf("Output array after Forward FFT, Normalization, and Inverse FFT :\n");
    for (auto& i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    /* free resources */
    cudaFree(d_data);


    cudaStreamDestroy(stream);

    cudaDeviceReset();

    return EXIT_SUCCESS;
}


//
//int main(int argc, char* argv[]) {
//    cufftHandle plan;
//    cudaStream_t stream = NULL;
//
//    int fft_size = 8;
//    int batch_size = 2;
//    int element_count = batch_size * fft_size;
//
//    using scalar_type = float;
//    using data_type = std::complex<scalar_type>;
//
//    std::vector<data_type> data(element_count, 0);
//
//    for (int i = 0; i < element_count; i++) {
//        data[i] = data_type(i, -i);
//    }
//
//    std::printf("Input array:\n");
//    for (auto& i : data) {
//        std::printf("%f + %fj\n", i.real(), i.imag());
//    }
//    std::printf("=====\n");
//
//    cufftComplex* d_data = nullptr;
//
//    CUFFT_CALL(cufftCreate(&plan));
//    CUFFT_CALL(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
//
//    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//    CUFFT_CALL(cufftSetStream(plan, stream));
//
//    // Create device data arrays
//    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_data), sizeof(data_type) * data.size()));
//    CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
//        cudaMemcpyHostToDevice, stream));
//
//    /*
//     * Note:
//     *  Identical pointers to data and output arrays implies in-place transformation
//     */
//    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
//
//    // Normalize the data
//    scaling_kernel << <1, 128, 0, stream >> > (d_data, element_count, 1.f / fft_size);
//
//    // The original data should be recovered after Forward FFT, normalization and inverse FFT
//    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));
//
//    CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
//        cudaMemcpyDeviceToHost, stream));
//
//    CUDA_RT_CALL(cudaStreamSynchronize(stream));
//
//    std::printf("Output array after Forward FFT, Normalization, and Inverse FFT :\n");
//    for (auto& i : data) {
//        std::printf("%f + %fj\n", i.real(), i.imag());
//    }
//    std::printf("=====\n");
//
//    /* free resources */
//    CUDA_RT_CALL(cudaFree(d_data))
//
//        CUFFT_CALL(cufftDestroy(plan));
//
//    CUDA_RT_CALL(cudaStreamDestroy(stream));
//
//    CUDA_RT_CALL(cudaDeviceReset());
//
//    return EXIT_SUCCESS;
//}

