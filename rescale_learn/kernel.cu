
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "array.h"
#include "rescale.h"
#include <cstddef>
#include <vector>
#include "kernel.cuh"
#include <thrust/device_vector.h>


int main()
{
	int nd = 16;
	int nt = 16;
	float seek_seconds = 0.0;
	int num_rescale_blocks = 2;
	float decay_timescale = 0.2; // Seconds?
	char ch;
	float thresh = 10.0;
	const char* out_filename = "fredda.cand";
	int cuda_device = 0;
	float kurt_thresh = 1e9;
	float std_thresh = 1e9;
	bool dump_data = false;
	float mean_thresh = 1e9;
	float dm0_thresh = 1e9;
	float cell_thresh = 1e9;
	int flag_grow = 3;
	int max_ncand_per_block = 4096;
	int mindm = 0;
	int maxbc = 32;
	int max_nblocks = INT_MAX;
	bool subtract_dm0 = false;

	

	
	
	// Load sigproc file
	

	int nbeams = 1;
	int nf = 4;
	size_t in_chunk_size = nbeams * nf * nt;

	// Create read buffer
	uint8_t* read_buf = (uint8_t*)malloc(sizeof(uint8_t) * in_chunk_size);


	

	/*array4d_t read_arr;
	read_arr.nw = 1;
	read_arr.nx = nt;
	read_arr.ny = nbeams;
	read_arr.nz = nf;*/

	array4d_t rescale_buf;
	rescale_buf.nw = nbeams;
	rescale_buf.nx = nf;
	rescale_buf.ny = 1;
	rescale_buf.nz = nt;
	array4d_malloc(&rescale_buf, dump_data, true);

	array4d_t out_buf;
	out_buf.nw = nbeams;
	out_buf.nx = 1;
	out_buf.ny = nd;
	out_buf.nz = nt;
	array4d_malloc(&out_buf, dump_data, true);

	double source_tsamp = 0.001728;
	// create rescaler
	rescale_gpu_t rescale;
	rescale.interval_samps = nt;
	rescale.target_mean = 0.0;
	//rescale.target_stdev = 1.0/sqrt((float) nf);
	rescale.target_stdev = 1.0;
	rescale.decay_constant = 0.35 * decay_timescale / source_tsamp; // This is how the_decimator.C does it, I think.
	rescale.mean_thresh = mean_thresh;
	rescale.std_thresh = std_thresh;
	rescale.kurt_thresh = kurt_thresh;
	rescale.flag_grow = flag_grow;
	rescale.dm0_thresh = dm0_thresh;
	rescale.cell_thresh = cell_thresh;
	// set guess of initial scale and offset to dm0 thresholding works
	printf("Rescaling to mean=%f stdev=%f decay constant=%f mean/std/kurtosis/dm0/Cell thresholds: %0.1f/%0.1f/%0.1f/%0.1f/%0.1f grow flags by %d channels\n",
		rescale.target_mean, rescale.target_stdev,
		rescale.decay_constant,
		rescale.mean_thresh, rescale.std_thresh, rescale.kurt_thresh,
		rescale.dm0_thresh, rescale.cell_thresh,
		rescale.flag_grow);
	//rescale_allocate(&rescale, nbeams*nf);
	rescale_allocate_gpu(&rescale, nbeams, nf, nt, true); // Need host memory allocated for rescale because we copy back to count flags
	if (num_rescale_blocks == 0)
	{
		rescale_set_scale_offset_gpu(&rescale, 1.0f, -128.0f); // Just pass it straight through without rescaling
	}
	else
	{
		rescale_set_scale_offset_gpu(&rescale, rescale.target_stdev / 18.0, -128.0f); // uint8 stdev is 18 and mean +128.
	}
	bool invert_freq = true;
	

	// creation of input array
	std::vector<uint8_t> vct_inpbuf_h(in_chunk_size);
	fill_inpVect(vct_inpbuf_h, nt, nf, invert_freq);
	thrust::device_vector<uint8_t> vct_inpbuf_d(vct_inpbuf_h.size());
	std::copy(vct_inpbuf_h.begin(), vct_inpbuf_h.end(), vct_inpbuf_d.begin());
	//!

	

	
	
	int num_flagged_beam_chans = 0;
	int num_flagged_times = 0;
	
	
		

		// File is in TBF order
		// Output needs to be BFT order
		// Do transpose and cast to float on the way through using GPU
		// copy raw data to state. Here we're a little dodgey

		
		uint8_t* read_buf_device = thrust::raw_pointer_cast(vct_inpbuf_d.data());		
		
		rescale_update_and_transpose_float_gpu(rescale, rescale_buf, read_buf_device, invert_freq, subtract_dm0);
		

    return 0;
}

void fill_inpVect(std::vector<uint8_t >& vct_inpbuf, const int  nt, const int nf, const bool  binvert_freq)
{
	uint8_t* piarr = (uint8_t*)malloc(nt * nf * sizeof(uint8_t));

	for (int iff = 0; iff < nf; ++iff)
	{
		uint8_t ia = (binvert_freq)? nf - iff : 1 + iff;
		ia *= 16;

		uint8_t istd = (binvert_freq) ?1<< (nf - iff) : 1<<(1 + iff);
		int ii = 1;

		for(int it = 0; it < nt; ++it)
		{
			piarr[iff * nt + it] = ia + ii * istd;
			ii = -ii;			
		}
	}

	uint8_t* ip = vct_inpbuf.data();
	for (int i = 0; i < nf; ++i)
	{
		for (int j = 0; j < nt; ++j)
		{
			ip[j * nf + i] = piarr[i * nt + j];
		}
	}
	free(piarr);
}