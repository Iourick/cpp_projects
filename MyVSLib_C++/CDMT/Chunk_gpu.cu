#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Chunk_gpu.cuh"

#include <vector>
#include "OutChunkHeader.h"

#include "Constants.h"
#include <chrono>

#include "helper_functions.h"
#include "helper_cuda.h"
#include <math_functions.h>
#include <complex>
#include "npy.hpp"
#include "TelescopeHeader.h"
#include "Clusterization.cuh"
#include "Statistical_preprocessing.cuh"


extern cudaError_t cudaStatus0 = cudaErrorInvalidDevice;


	size_t free_bytes, total_bytes;
	cudaError_t cuda_status = cudaMemGetInfo(&free_bytes, &total_bytes);

	extern const unsigned long long TOtal_GPU_Bytes = (long long)free_bytes;

	
    #define BLOCK_DIM 32
	CChunk_gpu::~CChunk_gpu()
	{
		if (m_pd_arrcoh_dm)
		{
			cudaFree(m_pd_arrcoh_dm);
		}
		

		if (m_pdcmpbuff_ewmulted)
		{
			cudaFree(m_pdcmpbuff_ewmulted);
		}

		if (m_pdbuff_rolled)
		{
			cudaFree(m_pdbuff_rolled);
		}

		if (m_pdoutImg)
		{
			cudaFree(m_pdoutImg);
		}

		cufftDestroy(m_fftPlanForward);
		cufftDestroy(m_fftPlanInverse);
	}
	//-----------------------------------------------------------
	CChunk_gpu::CChunk_gpu() :CChunkB()
	{
		m_pd_arrcoh_dm = nullptr;		
		m_pdbuff_rolled = nullptr;
		m_pdcmpbuff_ewmulted = nullptr;
		m_pdoutImg = nullptr;
	}
	//-----------------------------------------------------------

	CChunk_gpu::CChunk_gpu(const  CChunk_gpu& R) :CChunkB(R)
	{
		cudaMalloc(&m_pd_arrcoh_dm, R.m_coh_dm_Vector.size() * sizeof(double));
		cudaMemcpy(m_pd_arrcoh_dm, R.m_pd_arrcoh_dm, m_coh_dm_Vector.size() * sizeof(double), cudaMemcpyDeviceToDevice);

		m_Fdmt = R.m_Fdmt;

		checkCudaErrors(cudaMalloc((void**)&m_pdcmpbuff_ewmulted, R.m_nfft * R.m_nchan * R.m_npol / 2 * R.m_nbin * sizeof(cufftComplex)));

		checkCudaErrors(cudaMalloc((void**)&m_pdbuff_rolled, R.m_nfft * R.m_nchan * R.m_nbin * sizeof(float)));

		int msamp = get_msamp();
		checkCudaErrors(cudaMalloc((void**)&m_pdoutImg, msamp * R.m_len_sft * R.m_coh_dm_Vector.size() * sizeof(float)));
		cudaMemcpy(m_pdoutImg, R.m_pdoutImg, msamp * R.m_len_sft * R.m_coh_dm_Vector.size() * sizeof(float), cudaMemcpyDeviceToDevice);

		cufftDestroy(m_fftPlanForward);
		cufftDestroy(m_fftPlanInverse);
		create_fft_plans();

	}
	//-------------------------------------------------------------------

	CChunk_gpu& CChunk_gpu::operator=(const CChunk_gpu& R)
	{
		if (this == &R)
		{
			return *this;
		}
		CChunkB:: operator= (R);

		if (m_pd_arrcoh_dm)
		{
			cudaFree(m_pd_arrcoh_dm);
		}
		cudaMalloc(&m_pd_arrcoh_dm, R.m_coh_dm_Vector.size() * sizeof(double));
		cudaMemcpy(m_pd_arrcoh_dm, R.m_pd_arrcoh_dm, m_coh_dm_Vector.size() * sizeof(double), cudaMemcpyDeviceToDevice);
		
		if (m_pdcmpbuff_ewmulted)
		{
			cudaFree(m_pdcmpbuff_ewmulted);
		}

		
		checkCudaErrors(cudaMalloc((void**)&m_pdcmpbuff_ewmulted, R.m_nfft * R.m_nchan * R.m_npol / 2 * R.m_nbin * sizeof(cufftComplex)));

		if (m_pdbuff_rolled)
		{
			cudaFree(m_pdbuff_rolled);
		}
		checkCudaErrors(cudaMalloc((void**)&m_pdbuff_rolled, R.m_nfft * R.m_nchan *  R.m_nbin * sizeof(float)));

		if (m_pdoutImg)
		{
			cudaFree(m_pdoutImg);
		}
		int msamp = get_msamp();
		checkCudaErrors(cudaMalloc((void**)&m_pdoutImg, msamp * R.m_len_sft * R.m_coh_dm_Vector.size() * sizeof(float)));
		cudaMemcpy(m_pdoutImg, R.m_pdoutImg, msamp * R.m_len_sft * R.m_coh_dm_Vector.size() * sizeof(float), cudaMemcpyDeviceToDevice);

		m_Fdmt = R.m_Fdmt;
		cufftDestroy(m_fftPlanForward);
		cufftDestroy(m_fftPlanInverse);
		create_fft_plans();

		return *this;
	}
	//------------------------------------------------------------------
	CChunk_gpu::CChunk_gpu(
		const float Fmin
		, const float Fmax
		, const int npol
		, const int nchan
		, const unsigned int len_sft
		, const int Block_id
		, const int Chunk_id
		, const  float d_max
		, const  float d_min
		, const int ncoherent
		, const float sigma_bound
		, const int length_sum_wnd
		, const int nbin
		, const int nfft
		, const int noverlap
		, const float tsamp) : CChunkB(Fmin
			, Fmax
			, npol
			, nchan
			, len_sft
			, Block_id
			, Chunk_id
			, d_max
			, d_min
			, ncoherent
			, sigma_bound
			, length_sum_wnd
			, nbin
			, nfft
			, noverlap
			, tsamp)
	{
		// 1.
		const int ndm = m_coh_dm_Vector.size();
		// 1!

		cudaMalloc(&m_pd_arrcoh_dm, ndm * sizeof(double));
		cudaMemcpy(m_pd_arrcoh_dm, m_coh_dm_Vector.data(), ndm * sizeof(double), cudaMemcpyHostToDevice);		

		const int msamp = get_msamp();
		m_Fdmt = CFdmtGpu(
			m_Fmin
			, m_Fmax
			, m_nchan * m_len_sft // quant channels/rows of input image
			, msamp
			, m_len_sft
		);
		create_fft_plans();

		checkCudaErrors(cudaMalloc((void**)&m_pdcmpbuff_ewmulted, m_nfft * m_nchan * m_npol / 2 * m_nbin * sizeof(cufftComplex)));

		checkCudaErrors(cudaMalloc((void**)&m_pdbuff_rolled, m_nfft * m_nchan  * m_nbin * sizeof(float)));

		checkCudaErrors(cudaMalloc((void**)&m_pdoutImg, msamp * m_len_sft * m_coh_dm_Vector.size() * sizeof(float)));

	}

//-------------------------------------------------------------------------------------------------------
void CChunk_gpu::compute_chirp_channel()
{	
}
//--------------------------------------------------------------------------------------------
void CChunk_gpu::create_fft_plans()
{
	if (cufftPlanMany(&m_fftPlanForward,1, &m_nbin,
		NULL, 1, m_nbin, // *inembed, istride, idist
		NULL, 1, m_nbin, // *onembed, ostride, odist
		CUFFT_C2C, m_nfft * m_nchan * m_npol / 2) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return;
	}
	int mbin = get_mbin();
	
	checkCudaErrors(cufftPlanMany(&m_fftPlanInverse, 1, &mbin, NULL,            1,      mbin, NULL,           1,         mbin, CUFFT_C2C,  m_nfft * m_nchan * m_len_sft * m_npol / 2));
}


//------------------------------------------------------------------------------------------
__global__
void kernel_create_arr_bin_freqs_and_taper(double* d_parr_bin_freqs, double* d_parr_taper,  double  bw_chan,  int mbin)
{	
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= mbin)
	{
		return;
	}
	double temp = -0.5 * bw_chan + (ind + 0.5) * bw_chan / mbin;
	d_parr_bin_freqs[ind] = temp;
	d_parr_taper[ind] = 1.0 / sqrt(1.0 + pow(temp / (0.47 * bw_chan), 80));	
}
//----------------------------------------------------------------------------------------
__global__
void kernel_create_arr_freqs_chan(double* d_parr_freqs_chan, int len_sft, double bw_chan, double  Fmin, double bw_sub)
{
	int nchan = gridDim.y;
	int ichan = blockIdx.y;	
	int col_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (col_ind >= len_sft)
	{
		return;
	}
	 double freqs_sub = Fmin + bw_sub * (0.5 + ichan);
	 double vi = (double)(col_ind % len_sft);
	 double temp = bw_chan * (vi - len_sft / 2.0 + 0.5);
	d_parr_freqs_chan[ichan * len_sft + col_ind] = freqs_sub + temp;
}
//---------------------------------------------------

// INPUT:
//pcmparrRawSignalCur - raw signal, complex
// len = (m_header.m_npol / 2)*m_nfft * m_header.m_nchan  *  m_nbin
// pcmparrRawSignalCur - is complex-value matrix.
// rows = (m_header.m_npol / 2)*m_nfft * m_header.m_nchan;
// cols = m_nbin
// the first "m_nfft * m_header.m_nchan"  rows of the matrix "pcmparrRawSignalCur " correspond to polarization number 0
// the second, if "m_npol == 4", correspond to polarization number 1
// OUTPUT:
//pvctSuccessHeaders - is not being used, = nullptr
//  pvecImg - output image
	bool CChunk_gpu::process(void* pcmparrRawSignalCur
		, std::vector<COutChunkHeader>* pvctSuccessHeaders, std::vector<std::vector<float>>* pvecImg)
	{
		// 1. constant's calculation.
		const int mbin = get_mbin();   // bins after SFFT
		const int noverlap_per_channel = get_noverlap_per_channel();
		const int mbin_adjusted = get_mbin_adjusted(); // quantity of bins in time domain after unpadding
		const int msamp = get_msamp(); // bin's quantity of fdmt input in time domain (= fdmt cols)
		const int mchan = m_nchan * m_len_sft;// rows quantity of fdmt input
		// 1!

		// 2. Forward FFT execution
		// pcmparrRawSignalCur - input-output
		checkCudaErrors(cufftExecC2C(m_fftPlanForward, (cufftComplex*)pcmparrRawSignalCur, (cufftComplex*)pcmparrRawSignalCur, CUFFT_FORWARD));
		//2!	
		
		// 3. main loop.
		for (int idm = 0; idm < m_coh_dm_Vector.size(); ++idm)
		{

			// 4. Element wise multiplication - moving phases
			// OVERLOADED. 
			// implementation of this function is in "Chunk_fly_gpu.cu"
			// INPUT:
			//pcmparrRawSignalCur, size =  m_nfft * m_header.m_nchan * m_nbin * (m_header.m_npol / 2)
			//    can be represented as 4-dimentional array:  (m_header.m_npol / 2) x m_nfft x m_header.m_nchan x m_nbin 
			//	 idm - current number of dm		
			// OUTPUT:
			// m_pdcmpbuff_ewmulted, , size =  m_nfft * m_header.m_nchan * m_nbin * (m_header.m_npol / 2)
			//    can be represented as 4-dimentional array:  (m_header.m_npol / 2) x m_nfft x m_header.m_nchan x m_nbin 
			elementWiseMult(m_pdcmpbuff_ewmulted, (cufftComplex*)pcmparrRawSignalCur, idm);
			// !4

			// 5. Inverse FFT
			checkCudaErrors(cufftExecC2C(m_fftPlanInverse, m_pdcmpbuff_ewmulted, m_pdcmpbuff_ewmulted, CUFFT_INVERSE));
			// !5
			
			// 6. FFT - Normalization  plus summation of polarizations
			dim3 threads(1024, 1);
			dim3 blocks((m_nbin + threads.x - 1) / threads.x, m_nfft * m_nchan );
			//OUTPUT:
			// m_pdbuff_rolled - is intensity matrix, not yet unpadded, size =  m_nfft * m_nchan  * m_nbin		 
			roll_rows_normalize_sum_kernel << <blocks, threads >> > (m_pdbuff_rolled, m_pdcmpbuff_ewmulted, m_npol, m_nfft * m_nchan, m_nbin, m_nbin / 2);
			// !6
			
			// 7. Transposition submarices and unpadding
			// OUTPUT: fbuf is float matrix with dimensions: (m_nchan * m_len_sft) x msamp
			// INPUT:  m_pdbuff_rolled - can be interpreted as 4-dim matrix: m_nfft x m_nchan   x mbin x  m_len_sft
			// or (m_nfft x m_nchan ) matrices, each with dimension  mbin x m_len_sft written consequently in memory.
			// there are being done the following manipulations with this matrix:
			// 1. each of these  (m_nfft x m_nchan ) matrices is being transposed to  m_len_sft x mbin
			// 2. each of these resulting matrices is being unpadded from each side with "noverlap_per_channel"
			float* fbuf = (float*)m_pdcmpbuff_ewmulted;			
			dim3 threads_per_block1(512, 1, 1);
			dim3 blocks_per_grid1((mbin_adjusted + threads_per_block1.x - 1) / threads_per_block1.x, m_nchan * m_len_sft, m_nfft);
			transpose_unpadd_intensity << < blocks_per_grid1, threads_per_block1 >> > (fbuf, m_pdbuff_rolled, m_nfft, noverlap_per_channel
				, mbin_adjusted, m_nchan, m_len_sft, mbin);
			// !7

			// 8. calculation dedispersion shift for each row and perfoming corresponding shift
			float* parr_wfall_disp = m_pdbuff_rolled;
			double val_tsamp_wfall = (double)(m_len_sft * m_tsamp);
			double val_dm = m_coh_dm_Vector[idm];
			double f0 = (static_cast<double>(m_Fmax) - static_cast<double>(m_Fmin)) / (m_nchan * m_len_sft);
			dim3 threadsPerblock1(1024, 1, 1);
			dim3 blocksPerGrid1((msamp + threadsPerblock1.x - 1) / threadsPerblock1.x, m_nchan * m_len_sft, 1);
			dedisperse << <  blocksPerGrid1, threadsPerblock1 >> > (parr_wfall_disp, fbuf, val_dm
				, (double)m_Fmin, (double)m_Fmax, val_tsamp_wfall, f0, msamp);
			// !8
			
			// 9. Memory allocation for input to calculate fdmt-norm
			float* d_buff = reinterpret_cast<float*>(m_pdcmpbuff_ewmulted);
			float* d_parr_wfall_normInput = nullptr;
			cudaMalloc((void**)&d_parr_wfall_normInput, sizeof(float) * m_nchan * m_len_sft * msamp);
			// !9

			// 10. statistical preprocessing/cleaning input fdmt matrix
			// Explanations:
			// parr_wfall_disp - intensivity matrix, dimentions: m_nchan * m_len_sft x  msamp
			// d_parr_wfall_normInput - input matrix to calculate normalization-fdmt matrix.
			//   if some element of the matrix parr_wfall_disp is being excluded (=0), then 
			//  element of d_parr_wfall_normInput with the same index should be set to 0.0f.  Otherwise set to 1.0f
			// d_buff - auxillary memory buffer, size = 4 + 2 *  m_nchan * m_len_sft
			statistical_preprocessing::calc_fdmt_inputs(parr_wfall_disp, m_nchan * m_len_sft, msamp
				, d_buff, d_parr_wfall_normInput);
			// !10

			// 11. memory allocation for normalization fdmt matrix
			float* d_parr_wfall_norm = nullptr;
			cudaMalloc((void**)&d_parr_wfall_norm, msamp * m_len_sft * sizeof(float));
			// !11

			// 12. calculation normalization fdmt matrix
			m_Fdmt.process_image(d_parr_wfall_normInput, d_parr_wfall_norm, false);
			// !12

			// 13. fdmt calculation
			m_Fdmt.process_image(parr_wfall_disp, &m_pdoutImg[idm * msamp * m_len_sft], false);
			// !13
			
			// 14.normalization fdmt with fdmt-nomalization matrix:  fdmt = fdmt/sqrt(fdmt_norm)			
			fdmt_normalization << <(msamp * m_len_sft + 1024 - 1) / 1024, 1024 >> > 
				(&m_pdoutImg[idm * msamp * m_len_sft], d_parr_wfall_norm, msamp * m_len_sft, &m_pdoutImg[idm * msamp * m_len_sft]);			
			//!14
			cudaFree(d_parr_wfall_norm);
			cudaFree(d_parr_wfall_normInput);
		}
		
		 // 15. defining metrics array on host and device
		thrust::host_vector<int> h_bin_metrics(3);
		h_bin_metrics[0] = 2 * m_length_sum_wnd;
		h_bin_metrics[1] = 2 *m_length_sum_wnd;
		h_bin_metrics[2] = 4;

		thrust::device_vector<int> d_bin_metrics(h_bin_metrics.size());
		d_bin_metrics = h_bin_metrics;
		const int* d_pbin_metrics = thrust::raw_pointer_cast(d_bin_metrics.data());

		// 16. treshold	 --> GPU
		float* d_pTresh;   		
		cudaMalloc((void**)&d_pTresh, sizeof(float));
		cudaMemcpy(d_pTresh, &m_sigma_bound, sizeof(float), cudaMemcpyHostToDevice);
		
		// 17. Gathering candidates in an device array "d_parrCand"
		std::string filename = "output_";
		std::ostringstream oss;
		oss << filename << m_Block_id << "_ " << m_Chunk_id << ".log";
		filename = oss.str();
		const int QUantPower2_Wnd = 4;
		Cand* d_parrCand = nullptr;
		unsigned int h_quantCand = 0;
		clusterization::gather_candidates_in_dynamicalArray(m_pdoutImg // fdmt output image, input
			, m_len_sft* m_coh_dm_Vector.size() // quant rows of m_pdoutImg, input 
			, msamp // quant columns of m_pdoutImg, input 
			, *d_pTresh // device allocated detection's threshold , input 
			, QUantPower2_Wnd // windows{pow(2,0), pow(2,1),.., pow(2,QUantPower2_Wnd-1)}, input 
			, &d_parrCand // array with candidates, device allocated, structure Cand is declared in Candidate.cuh
			                           // output
			, &h_quantCand// size of d_parrCand, device allocated variable, output
		);

		cudaFree(d_parrCand);

		/*----------------------------------------------------------------*/
		/*----------------------------------------------------------------*/
		/*------ OPTION FOR CLUSTERIZATION ----------------------------------------------------------*/
		/*----------------------------------------------------------------*/
		/*----------------------------------------------------------------*/

		// defining clasterization's splitters
		
	
	thrust::host_vector<int> h_bin_splitters(3);
		h_bin_splitters[0] = 2;
		h_bin_splitters[1] = 1;
		h_bin_splitters[2] = 2;

		thrust::device_vector<int> d_bin_splitters(h_bin_splitters.size());
		d_bin_splitters = h_bin_splitters;
		//!
		std::string outfile{ "candidates.log" };
		const float t_chunkBegin = 0.0; // time of beginning the chunk
		clusterization::clusterization_main(m_pdoutImg // fdmt output image, input
			, m_len_sft* m_coh_dm_Vector.size() // quant rows of m_pdoutImg, input 
			, msamp // quant columns of m_pdoutImg, input 
			, *d_pTresh // device allocated detection's threshold , input 
			, QUantPower2_Wnd // windows{1,2,..,QUantPower2_Wnd-1}, input 
			,thrust::raw_pointer_cast(d_bin_splitters.data())
			, outfile
			, m_tsamp *m_len_sft // array with candidates, device allocated, structure Cand is declared in Candidate.cuh
			,t_chunkBegin
		); 
		
		
		/*----------------------------------------------------------------------------------------------------*/
		/*----------------------------------------------------------------------------------------------------*/
		/*---------  this code is being used for visualization in my application the only -----------------------------*/
		/*----------------------------------------------------------------------------------------------------*/
		/*----------------------------------------------------------------------------------------------------*/

		std::vector<std::vector<float>>* pvecImg_temp = nullptr;
		if (nullptr != pvecImg)
		{
			pvecImg->resize(m_coh_dm_Vector.size());
			for (auto& row : *pvecImg)
			{
				row.resize(msamp * m_len_sft);
			}
			pvecImg_temp = pvecImg;
		}
		else
		{
			pvecImg_temp = new std::vector<std::vector<float>>;
			pvecImg_temp->resize(m_coh_dm_Vector.size());
			for (auto& row : *pvecImg_temp)
			{
				row.resize(msamp * m_len_sft);
			}
		}
		for (int i = 0; i < m_coh_dm_Vector.size(); ++i)
		{
			cudaMemcpy(pvecImg_temp->at(i).data(), &m_pdoutImg[i * msamp * m_len_sft], msamp * m_len_sft * sizeof(float), cudaMemcpyDeviceToHost);
		}
		/*--------- !  this code is for visualization in my application----------------------------------------------------------------*/ 

		return true;
	}
	//---------------------------------------------------------------------------------------
	// INPUT:
	//arr - matrix with dimentions:  (npol *rows) x cols
	// shift - shifting value, in our case shift = cols/2
	// npol - polarizations, = 2 OR 4
	// OUTPUT:
	// arr_rez - matrix of intesivities, dementions: rows x cols
	//Functionality:
	// 1. shifts every row
	// 2. calculates intensivity: re*re + im * im for each element
	// 3. if npol == 4 adds matrices M0+M1 for polarization 0 and 1
	// 4. fft-normalization, dividing on "cols"
	// Sample for call:
	//dim3 threads(1024, 1);
	//dim3 blocks((m_nbin + threads.x - 1) / threads.x, m_nfft* m_nchan);
	__global__ void roll_rows_normalize_sum_kernel(float* arr_rez, cufftComplex* arr, const int npol, const int rows
		, const  int cols, const  int shift)
	{
		int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx0 >= cols)
		{
			return;
		}
		int ind_new = blockIdx.y * cols + (idx0 + shift) % cols;
		int ind = blockIdx.y * cols + idx0;
		arr_rez[ind_new] = 0.0f;
		for (int ipol = 0; ipol < npol / 2; ++ipol)
		{
			double x = static_cast<double>(arr[ind].x);
			double y = static_cast<double>(arr[ind].y);
			arr_rez[ind_new] += static_cast<float>((x * x + y * y) / cols / cols);
			ind += rows * cols;
		}
	}
//------------------------------------------------------------------

__global__ 	void  transpose_unpadd_intensity(float* arout, float* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, int mbin)
{
	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(ibin < mbin_adjusted))
	{
		return;
	}
	int ifft = blockIdx.z;
	int isub = blockIdx.y / nchan;
	int ichan = blockIdx.y % nchan;
	int ibin_adjusted = ibin + noverlap_per_channel;
	int isamp = ibin + mbin_adjusted * ifft;
	int msamp = mbin_adjusted * nfft;

	// Select bins from valid region and reverse the frequency axis				
	int iinp = ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
	int iout = (isub * nchan + nchan - ichan - 1) * msamp + isamp;	
	arout[iout] = arin[iinp];
}
//------------------------------------------------------------
__global__
void calc_intensity_kernel  (float *intensity, const int len, const int npol, cufftComplex* fbuf)
{
	 int ind  = blockIdx.x * blockDim.x + threadIdx.x;
	 if (ind < len)
	 {
		 intensity[ind] = 0.0f;
		 for (int i = 0; i < npol / 2; ++i)
		 {
			 cufftComplex* p = &fbuf[i * len + ind];
			 intensity[ind] += (*p).x * (*p).x + (*p).y * (*p).y;
		 }
	 }
}
//----------------------------------------------------------------------------------------
// #define BLOCK_DIM 32//16
//dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);
//dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
//https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
__global__ void transpose(float* odata, float* idata, int width, int height)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

	// read the matrix tile into shared memory
		// load one element per thread from device memory (idata) and store it
		// in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	// synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}
//------------------------------------------
__global__
void calcPowerMtrx_kernel(float* output, const int height, const int width, const int npol, cufftComplex* input)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile
	unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	if ((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;	

		float sum = 0.;
		for (int i = 0; i < npol / 2; ++i)
		{
			sum += fnc_norm2(&input[i * height * width + index_in]);
		}

		tile[threadIdx.y][threadIdx.x] = sum;
	}

	// synchronise to ensure all writes to block[][] have completed
	__syncthreads();
   
	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	if ((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		output[index_out] = tile[threadIdx.x][threadIdx.y];
		
	}
}

//-----------------------------------------------------------------------------

// INPUT:
// parrIntesity - matrix with dimensions = (blocksPerGrid1.y) x cols
// -dm - currtent dispersion measure
// f_min, f_max - bounding frequensies
// val_tsamp_wfall - value of time-bin
// OUTPUT:
// parr_wfall_disp  - matrix with dimensions = (blocksPerGrid1.y) x cols
// functionality:
// 1. for each row of input matrix is being calculated integer value for shifting - "ishift"
// 2. row is being shifted in accordance with "ishift"
// sample for call:
//dim3 threadsPerblock1(1024, 1, 1);
//dim3 blocksPerGrid1((msamp + threadsPerblock1.x - 1) / threadsPerblock1.x, m_nchan* m_len_sft, 1);
__global__
void dedisperse(float*parr_wfall_disp, float* parrIntesity, double dm,  double f_min, double f_max, double   val_tsamp_wfall, double foff, int cols)
{
 int idx0 =  blockIdx.x * blockDim.x + threadIdx.x;
 if (!(idx0 < cols))
 {
	 return;
 }
 
 int nchans = gridDim.y;
 int ichan = blockIdx.y;
 double temp = (double)(nchans - 1 - ichan) * foff + f_min;
 double temp1 = 4.148808e3 * dm * (1.0 / (f_min * f_min) - 1.0 / (temp * temp));
 int ishift = round(temp1 / val_tsamp_wfall);
 
 int ind_new = blockIdx.y * cols + (idx0 + ishift) % cols;
 int ind = blockIdx.y * cols + idx0;
 //
 parr_wfall_disp[ind_new] =  parrIntesity[ind]; 
}


__global__	void  transpose_unpadd_kernel(cufftComplex* fbuf, cufftComplex* arin, int nfft, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, int mbin)
{
	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(ibin < mbin_adjusted))
	{
		return;
	}
	int ipol = blockIdx.z / nfft;
	int ifft = blockIdx.z % nfft;
	int isub = blockIdx.y / nchan;
	int ichan = blockIdx.y % nchan;
	int ibin_adjusted = ibin + noverlap_per_channel;
	int isamp = ibin + mbin_adjusted * ifft;
	int msamp = mbin_adjusted * nfft;
	// Select bins from valid region and reverse the frequency axis		
   // printf("ipol = %i   ifft =  %i\n", ipol, ifft);
	int iinp = ipol * nfft * nsub * nchan * mbin + ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
	int iout = ipol * msamp * nsub * nchan + isamp * nsub * nchan + isub * nchan + nchan - ichan - 1;
	// Select bins from valid region and reverse the frequency axis		

	fbuf[iout].x = arin[iinp].x;
	fbuf[iout].y = arin[iinp].y;

}
//-------------------------------------------------------------------
__global__	void  transpose_unpadd_kernel_(float* fbuf, cufftComplex* arin, int nfft, int npol, int noverlap_per_channel
	, int mbin_adjusted, const int nsub, const int nchan, const int mbin)
{
	int  ibin = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(ibin < mbin_adjusted))
	{
		return;
	}
	
	int ifft = blockIdx.z ;
	int isub = blockIdx.y / nchan;
	int ichan = blockIdx.y % nchan;
	int ibin_adjusted = ibin + noverlap_per_channel;
	int isamp = ibin + mbin_adjusted * ifft;
	int msamp = mbin_adjusted * nfft;
	// Select bins from valid region and reverse the frequency axis		
   // printf("ipol = %i   ifft =  %i\n", ipol, ifft);
	int iinp =  ifft * nsub * nchan * mbin + (nsub - isub - 1) * nchan * mbin + ichan * mbin + ibin_adjusted;
	int iout =  isamp * nsub * nchan + isub * nchan + nchan - ichan - 1;
	// Select bins from valid region and reverse the frequency axis		
	fbuf[iout] = 0.0f;
	for (int ipol = 0; ipol < npol / 2; ++ipol)
	{
		iinp += nfft * nsub * nchan * mbin;
		iout += msamp * nsub * nchan;
		fbuf[iout] += arin[iinp].x * arin[iinp].x + arin[iinp].y * arin[iinp].y;
	}

}
	//-----------------------------------------------------------------------
	__global__ void  divide_cufftComplex_array_kernel(cufftComplex* d_arr, int len, float val)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= len)
		{
			return;
		}
		d_arr[idx].x /= val;
		d_arr[idx].y /= val;
	}
//-------------------------------------------------------------------------------------------------
	__global__
		void scaling_kernel(cufftComplex* data, long long element_count, float scale)
	{
		const int tid = threadIdx.x;
		const int stride = blockDim.x;
		for (long long i = tid; i < element_count; i += stride)
		{
			data[i].x *= scale;
			data[i].y *= scale;
		}
	}
	//-----------------------------------------------------
__global__ void  element_wise_cufftComplex_mult_kernel(cufftComplex * d_arrOut, cufftComplex * d_arrInp0, cufftComplex * d_arrInp1
	, int npol, int nfft, int dim2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dim2)
	{
		return;
	}

	int ipol = blockIdx.z;
	int ifft = blockIdx.y;
	int ibegin = ipol * nfft * dim2 + ifft * dim2;
	d_arrOut[ibegin + idx].x = d_arrInp0[ibegin + idx].x* d_arrInp1[idx].x - d_arrInp0[ibegin + idx].y * d_arrInp1[idx].y;
	d_arrOut[ibegin + idx].y = d_arrInp0[ibegin + idx].x* d_arrInp1[idx].y + d_arrInp0[ibegin + idx].y * d_arrInp1[idx].x;
	
}
//------------------------------------------------------------------------------------
void CChunk_gpu ::elementWiseMult(cufftComplex* d_arrOut, cufftComplex* d_arrInp0, int  idm)
{
	
 }
//--------------------------------------------------------------------------------

__global__ void roll_rows_and_normalize_kernel(cufftComplex* arr_rez, cufftComplex* arr, int rows, int cols, int shift)
{

	int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx0 >= cols)
	{
		return;
	}
	int ind_new = blockIdx.y * cols + (idx0 + shift) % cols;
	int ind = blockIdx.y * cols + idx0;
	arr_rez[ind_new].x = arr[ind].x / cols;
	arr_rez[ind_new].y = arr[ind].y / cols;

}
//--------------------------------------
void CChunk_gpu::set_chunkid(const int nC)
{
	m_Chunk_id = nC;
}
//--------------------------------------
void CChunk_gpu::set_blockid(const int nC)
{
	m_Block_id = nC;
}
//-------------------------------------------------------------------
__device__
float fnc_norm2(cufftComplex* pc)
{	
	return ((*pc).x * (*pc).x + (*pc).y * (*pc).y);
}
//----------------------------------------------------
__global__
void calcMultiTransposition_kernel(fdmt_type_* output, const int height, const int width, fdmt_type_* input)
{
	__shared__ fdmt_type_ tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int ichan = blockIdx.z;
	// Transpose data from global to shared memory
	if (x < width && y < height)
	{
		tile[threadIdx.y][threadIdx.x] = input[ichan * height * width + y * width + x];
	}
	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width)
	{
		output[ichan * height * width + y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}
//------------------------------------------
__global__
void calcPartSum_kernel(float* d_parr_out, const int lenChunk, const int npol_physical, cufftComplex* d_parr_inp)
{
	int ichan = blockIdx.y;
	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind < lenChunk)
	{
		float sum = 0;
		for (int i = 0; i < npol_physical; ++i)
		{
			sum += fnc_norm2(&d_parr_inp[(ichan * npol_physical + i) * lenChunk + ind]);
		}
		d_parr_out[ichan * lenChunk + ind] = sum;
	}
}

//------------------------------------------
__global__
void multiTransp_kernel(float* output, const int height, const int width, float* input)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Shared memory tile

	int numchan = blockIdx.z;
	float* pntInp = &input[numchan * height * width];
	float* pntOut = &output[numchan * height * width];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	// Transpose data from global to shared memory
	if (x < width && y < height) {
		tile[threadIdx.y][threadIdx.x] = pntInp[y * width + x];
	}

	__syncthreads();

	// Calculate new indices for writing to output
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// Transpose data from shared to global memory
	if (x < height && y < width) {
		pntOut[y * height + x] = tile[threadIdx.x][threadIdx.y];
	}
}


//----------------------------------------------------
__global__
void fdmt_normalization(fdmt_type_* d_arr, fdmt_type_* d_norm, const int len, float* d_pOutArray)
{

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= len)
	{
		return;
	}
	d_pOutArray[idx] = ((float)d_arr[idx]) / sqrtf(((float)d_norm[idx]) + 1.0E-8);

}








