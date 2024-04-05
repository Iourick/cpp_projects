﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <array>
#include <iostream>
#include <string>

#include <vector>
//#include <unistd.h>
#include <math.h>
#include <time.h>
#include<errno.h>
#include <cuda.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
//#include <getopt.h>
#include <hdf5.h>
#include <stdlib.h>
//#include <cstddef>  // For C++
//#include <cstdint>
//#include <cstdlib>



#define HEADERSIZE 4096
#define DMCONSTANT 2.41e-10

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
// Struct for header information
struct header {
    int64_t headersize, buffersize;
    unsigned int nchan, nsamp, nbit, nif, nsub;
    int machine_id, telescope_id, nbeam, ibeam, sumif;
    double tstart, tsamp, fch1, foff, fcen, bwchan;
    double src_raj, src_dej, az_start, za_start;
    char source_name[80], ifstream[8], inpfile[80];
    char* rawfname[4];
};

struct header read_h5_header(char* fname);
void get_channel_chirp(double fcen, double bw, float dm, int nchan, int nbin, int nsub, cufftComplex* c);
__global__ void transpose_unpadd_and_detect(cufftComplex* cp1, cufftComplex* cp2, int nbin, int nchan, int nfft, int nsub, int noverlap, int nsamp, float* fbuf);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b);
static __global__ void PointwiseComplexMultiply(cufftComplex* a, cufftComplex* b, cufftComplex* c, int nx, int ny, int l, float scale);
__global__ void unpack_and_padd(char* dbuf0, char* dbuf1, char* dbuf2, char* dbuf3, int nsamp, int nbin, int nfft, int nsub, int noverlap, cufftComplex* cp1, cufftComplex* cp2);
__global__ void swap_spectrum_halves(cufftComplex* cp1, cufftComplex* cp2, int nx, int ny);
__global__ void compute_chirp(double fcen, double bw, float* dm, int nchan, int nbin, int nsub, int ndm, cufftComplex* c);
__global__ void compute_block_sums(float* z, int nchan, int nblock, int nsum, float* bs1, float* bs2);
__global__ void compute_channel_statistics(int nchan, int nblock, int nsum, float* bs1, float* bs2, float* zavg, float* zstd);
__global__ void redigitize(float* z, int nchan, int nblock, int nsum, float* zavg, float* zstd, float zmin, float zmax, unsigned char* cz);
__global__ void decimate_and_redigitize(float* z, int ndec, int nchan, int nblock, int nsum, float* zavg, float* zstd, float zmin, float zmax, unsigned char* cz);
void write_filterbank_header(struct header h, FILE* file);
__global__ void kernel_Sum_cols(unsigned char* d_inp, const int NRows, const int NCols, int* d_arrOut);
int readHeader5InBinary(const char* strFilePath, header& Head);
int writeOptions(const char* strFilePath, const int noverlap, const int nbin, const int ndec, const char* obsid
    , const int part, const int device);
int readOptions(const char* strFilePath, int& noverlap, int& nbin, int& ndec, char* obsid
    , int& part, int& device);

// Usage
void usage()
{
    printf("cdmt -P <part> -d <DM start,step,num> -D <GPU device> -b <ndec> -N <forward FFT size> -n <overlap region> -o <outputname> <file.h5>\n\n");
    printf("Compute coherently dedispersed SIGPROC filterbank files from LOFAR complex voltage data in HDF5 format.\n");
    printf("-P <part>        Specify part number for input file [integer, default: 0]\n");
    printf("-D <GPU device>  Select GPU device [integer, default: 0]\n");
    printf("-b <ndec>        Number of time samples to average [integer, default: 1]\n");
    printf("-d <DM start, step, num>  DM start and stepsize, number of DM trials\n");
    printf("-o <outputname>           Output filename [default: cdmt]\n");
    printf("-N <forward FFT size>     Forward FFT size [integer, default: 65536]\n");
    printf("-n <overlap region>       Overlap region [integer, default: 2048]\n");

    return;
}

int main(int argc, char* argv[])
{
    int i, nsamp, nfft, mbin, nvalid, nchan = 8, nbin = 65536, noverlap = 2048, nsub = 20, ndm, ndec = 1;
    int idm, iblock, nread, mchan, msamp, mblock, msum = 1024;
    char* header, * h5buf[4], * dh5buf[4];
    FILE* rawfile[4], * file;
    unsigned char* cbuf, * dcbuf;
    float* fbuf, * dfbuf;
    float* bs1, * bs2, * zavg, * zstd;
    cufftComplex* cp1, * cp2, * dc, * cp1p, * cp2p;
    cufftHandle ftc2cf, ftc2cb;
    int idist, odist, iembed, oembed, istride, ostride;
    dim3 blocksize, gridsize;
    struct header h5;
    clock_t startclock;
    float* dm, * ddm, dm_start, dm_step;
    char fname[128], fheader[1024], * h5fname, obsid[128] = "cdmt";
    int bytes_read;
    int part = 0, device = 0;
    int arg = 0;
    FILE** outfile;
    /*
    // Read options
    if (argc > 1) {
        while ((arg = getopt(argc, argv, "P:d:D:ho:b:N:n:")) != -1) {
            switch (arg) {

            case 'n':
                noverlap = atoi(optarg);
                break;

            case 'N':
                nbin = atoi(optarg);
                break;

            case 'b':
                ndec = atoi(optarg);
                break;

            case 'o':
                strcpy(obsid, optarg);
                break;

            case 'P':
                part = atoi(optarg);
                break;

            case 'D':
                device = atoi(optarg);
                break;

            case 'd':
                sscanf(optarg, "%f,%f,%d", &dm_start, &dm_step, &ndm);
                break;

            case 'h':
                usage();
                return 0;
            }
        }
    }
    else {
        usage();
        return 0;
    }
    h5fname = argv[optind];
    */
    // Read HDF5 header
    //h5 = read_h5_header(h5fname); !!
    //readOptions("Options.bin", noverlap, nbin, ndec, obsid, part, device);
    // cdmt -n 8192 -N 32768 -P 001 -b 4 -d 0.5,1.0,80 -o tmp/test L2012176_SAP000_B000_S0_P001_bf.h5
    //

    h5fname = new char[200];
    memset(h5fname, 0, 200 * sizeof(char));
    strcpy(h5fname, "D://BASSA//hdf5_data//L2012176_SAP000_B000_S0_P001_bf.h5");
    h5 = read_h5_header(h5fname);
    dm_start = 0.5;
    dm_step = 1.0;
    ndm = 80;
    noverlap = 8192;
    nbin = 32768;
    ndec = 4;
    part = 0;
    device = 0;
    strcpy(obsid, "tmp/test");
    //readHeader5InBinary("Header5File.bin", h5);

    // Set number of subbands
    nsub = h5.nsub;

    // Adjust header for filterbank format
    h5.tsamp *= nchan * ndec;
    h5.nchan = nsub * nchan;
    h5.nbit = 8;
    h5.fch1 = h5.fcen + 0.5 * h5.nsub * h5.bwchan - 0.5 * h5.bwchan / nchan;
    h5.foff = -fabs(h5.bwchan / nchan);

    // Data size
    nvalid = nbin - 2 * noverlap;
    nsamp = 100 * nvalid;
    nfft = (int)ceil(nsamp / (float)nvalid);
    mbin = nbin / nchan;
    mchan = nsub * nchan;
    msamp = nsamp / nchan;
    mblock = msamp / msum;

    printf("nbin: %d nfft: %d nsub: %d mbin: %d nchan: %d nsamp: %d nvalid: %d\n", nbin, nfft, nsub, mbin, nchan, nsamp, nvalid);

    // Set device
    checkCudaErrors(cudaSetDevice(device));

    // Allocate memory for complex timeseries
    checkCudaErrors(cudaMalloc((void**)&cp1, sizeof(cufftComplex) * nbin * nfft * nsub));
    checkCudaErrors(cudaMalloc((void**)&cp2, sizeof(cufftComplex) * nbin * nfft * nsub));
    checkCudaErrors(cudaMalloc((void**)&cp1p, sizeof(cufftComplex) * nbin * nfft * nsub));
    checkCudaErrors(cudaMalloc((void**)&cp2p, sizeof(cufftComplex) * nbin * nfft * nsub));

    // Allocate device memory for chirp
    checkCudaErrors(cudaMalloc((void**)&dc, sizeof(cufftComplex) * nbin * nsub * ndm));

    // Allocate device memory for block sums
    checkCudaErrors(cudaMalloc((void**)&bs1, sizeof(float) * mblock * mchan));
    checkCudaErrors(cudaMalloc((void**)&bs2, sizeof(float) * mblock * mchan));

    // Allocate device memory for channel averages and standard deviations
    checkCudaErrors(cudaMalloc((void**)&zavg, sizeof(float) * mchan));
    checkCudaErrors(cudaMalloc((void**)&zstd, sizeof(float) * mchan));

    // Allocate memory for redigitized output and header
    header = (char*)malloc(sizeof(char) * HEADERSIZE);
    for (i = 0; i < 4; i++) {
        h5buf[i] = (char*)malloc(sizeof(char) * nsamp * nsub);
        checkCudaErrors(cudaMalloc((void**)&dh5buf[i], sizeof(char) * nsamp * nsub));
    }

    // Allocate output buffers
    fbuf = (float*)malloc(sizeof(float) * nsamp * nsub);
    checkCudaErrors(cudaMalloc((void**)&dfbuf, sizeof(float) * nsamp * nsub));
    cbuf = (unsigned char*)malloc(sizeof(unsigned char) * msamp * mchan / ndec);
    checkCudaErrors(cudaMalloc((void**)&dcbuf, sizeof(unsigned char) * msamp * mchan / ndec));

    // Allocate DMs and copy to device
    dm = (float*)malloc(sizeof(float) * ndm);
    for (idm = 0; idm < ndm; idm++)
        dm[idm] = dm_start + (float)idm * dm_step;
    checkCudaErrors(cudaMalloc((void**)&ddm, sizeof(float) * ndm));
    checkCudaErrors(cudaMemcpy(ddm, dm, sizeof(float) * ndm, cudaMemcpyHostToDevice));

    // Generate FFT plan (batch in-place forward FFT)
    idist = nbin;  odist = nbin;  iembed = nbin;  oembed = nbin;  istride = 1;  ostride = 1;
    checkCudaErrors(cufftPlanMany(&ftc2cf, 1, &nbin, &iembed, istride, idist, &oembed, ostride, odist, CUFFT_C2C, nfft * nsub));

    // Generate FFT plan (batch in-place backward FFT)
    idist = mbin;  odist = mbin;  iembed = mbin;  oembed = mbin;  istride = 1;  ostride = 1;
    checkCudaErrors(cufftPlanMany(&ftc2cb, 1, &mbin, &iembed, istride, idist, &oembed, ostride, odist, CUFFT_C2C, nchan * nfft * nsub));

    // Compute chirp
    blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
    gridsize.x = nsub / blocksize.x + 1; gridsize.y = nchan / blocksize.y + 1; gridsize.z = ndm / blocksize.z + 1;
    compute_chirp << <gridsize, blocksize >> > (h5.fcen, nsub * h5.bwchan, ddm, nchan, nbin, nsub, ndm, dc);

    // Write temporary filterbank header
    FILE*file0 = fopen("tmp/header.fil", "w");// ОТЛИЧИЕ В ОТКРЫТИИ ФАЙЛА!!
    write_filterbank_header(h5, file0);
    fclose(file0);
    file = fopen("tmp/header.fil", "r");
    bytes_read = fread(fheader, sizeof(char), 1024, file);
    fclose(file);

    // Format file names and open
    outfile = (FILE**)malloc(sizeof(FILE*) * ndm);
    for (idm = 0; idm < ndm; idm++) {
        sprintf(fname, "%s_cDM%06.2f_P%03d.fil", obsid, dm[idm], part);

        outfile[idm] = fopen(fname, "w");
    }

    // Write headers
    for (idm = 0; idm < ndm; idm++) {
        // Send header
        fwrite(fheader, sizeof(char), bytes_read, outfile[idm]);
    }

    // Read files
    for (i = 0; i < 4; i++) {
        rawfile[i] = fopen(h5.rawfname[i], "rb");
        
    }

    // Loop over input file contents
    for (iblock = 0;; iblock++)
    {
        // Read block
        startclock = clock();
        for (i = 0; i < 4; i++)
        {
            //nread = fread(h5buf[i], sizeof(char), nsamp * nsub, rawfile[i]) / nsub;
            nread = fread(h5buf[i], sizeof(char), nsamp * nsub, rawfile[i]);
            nread = nread / nsub;
            std::cout << "nread =" << nread << std::endl;
        }
        if (nread == 0)
            break;
        printf("Block: %d: Read %d MB in %.2f s\n", iblock, sizeof(char) * nread * nsub * 4 / (1 << 20), (float)(clock() - startclock) / CLOCKS_PER_SEC);

        // Copy buffers to device
        startclock = clock();
        for (i = 0; i < 4; i++)
        {
            checkCudaErrors(cudaMemcpy(dh5buf[i], h5buf[i], sizeof(char) * nread * nsub, cudaMemcpyHostToDevice));
        }
        // Unpack data and padd data
        blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
        gridsize.x = nbin / blocksize.x + 1; gridsize.y = nfft / blocksize.y + 1; gridsize.z = nsub / blocksize.z + 1;
        unpack_and_padd << <gridsize, blocksize >> > (dh5buf[0], dh5buf[1], dh5buf[2], dh5buf[3], nread, nbin, nfft, nsub, noverlap, cp1p, cp2p);

        // Perform FFTs
        checkCudaErrors(cufftExecC2C(ftc2cf, (cufftComplex*)cp1p, (cufftComplex*)cp1p, CUFFT_FORWARD));
        checkCudaErrors(cufftExecC2C(ftc2cf, (cufftComplex*)cp2p, (cufftComplex*)cp2p, CUFFT_FORWARD));

        // Swap spectrum halves for large FFTs
        blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
        gridsize.x = nbin / blocksize.x + 1; gridsize.y = nfft * nsub / blocksize.y + 1; gridsize.z = 1;
        swap_spectrum_halves << <gridsize, blocksize >> > (cp1p, cp2p, nbin, nfft * nsub);

        // Loop over dms
        for (idm = 0; idm < ndm; idm++) {

            // Perform complex multiplication of FFT'ed data with chirp
            blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
            gridsize.x = nbin * nsub / blocksize.x + 1; gridsize.y = nfft / blocksize.y + 1; gridsize.z = 1;
            PointwiseComplexMultiply << <gridsize, blocksize >> > (cp1p, dc, cp1, nbin * nsub, nfft, idm, 1.0 / (float)nbin);
            PointwiseComplexMultiply << <gridsize, blocksize >> > (cp2p, dc, cp2, nbin * nsub, nfft, idm, 1.0 / (float)nbin);

            // Swap spectrum halves for small FFTs
            blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
            gridsize.x = mbin / blocksize.x + 1; gridsize.y = nchan * nfft * nsub / blocksize.y + 1; gridsize.z = 1;
            swap_spectrum_halves << <gridsize, blocksize >> > (cp1, cp2, mbin, nchan * nfft * nsub);

            // Perform FFTs
            checkCudaErrors(cufftExecC2C(ftc2cb, (cufftComplex*)cp1, (cufftComplex*)cp1, CUFFT_INVERSE));
            checkCudaErrors(cufftExecC2C(ftc2cb, (cufftComplex*)cp2, (cufftComplex*)cp2, CUFFT_INVERSE));

            // Detect data
            blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
            gridsize.x = mbin / blocksize.x + 1; gridsize.y = nchan / blocksize.y + 1; gridsize.z = nfft / blocksize.z + 1;
            transpose_unpadd_and_detect << <gridsize, blocksize >> > (cp1, cp2, mbin, nchan, nfft, nsub, noverlap / nchan, nread / nchan, dfbuf);

            // Compute block sums for redigitization
            blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
            gridsize.x = mchan / blocksize.x + 1; gridsize.y = mblock / blocksize.y + 1; gridsize.z = 1;
            compute_block_sums << <gridsize, blocksize >> > (dfbuf, mchan, mblock, msum, bs1, bs2);

            // Compute channel stats
            blocksize.x = 32; blocksize.y = 1; blocksize.z = 1;
            gridsize.x = mchan / blocksize.x + 1; gridsize.y = 1; gridsize.z = 1;
            compute_channel_statistics << <gridsize, blocksize >> > (mchan, mblock, msum, bs1, bs2, zavg, zstd);

            // Redigitize data to 8bits
            blocksize.x = 32; blocksize.y = 32; blocksize.z = 1;
            gridsize.x = mchan / blocksize.x + 1; gridsize.y = mblock / blocksize.y + 1; gridsize.z = 1;
            if (ndec == 1)
                redigitize << <gridsize, blocksize >> > (dfbuf, mchan, mblock, msum, zavg, zstd, 3.0, 5.0, dcbuf);
            else
                decimate_and_redigitize << <gridsize, blocksize >> > (dfbuf, ndec, mchan, mblock, msum, zavg, zstd, 3.0, 5.0, dcbuf);

            //********************************************************************************************
            //********************************************************************************************
            //********************************************************************************************
            int* dcbuf_arr = NULL;
            checkCudaErrors(cudaMalloc((void**)&dcbuf_arr, sizeof(int) * mchan));
            int blocksPerGrid = mchan;
            unsigned int treadsPerBlock = 256;
            std::size_t sz = treadsPerBlock * sizeof(int);
            kernel_Sum_cols << < blocksPerGrid, treadsPerBlock, sz >> > (dcbuf, mchan, msamp / ndec, dcbuf_arr);
            checkCudaErrors(cudaMemcpy(cbuf, dcbuf, sizeof(int) * mchan, cudaMemcpyDeviceToHost));
            fwrite(cbuf, sizeof(int), mchan, outfile[idm]);
            cudaFree(dcbuf_arr);
            //********************************************************************************************
            //********************************************************************************************
            //********************************************************************************************

            //// Copy buffer to host
            //checkCudaErrors(cudaMemcpy(cbuf,dcbuf,sizeof(unsigned char)*msamp*mchan/ndec,cudaMemcpyDeviceToHost));

            //// Write buffer
            //fwrite(cbuf,sizeof(char),nread*nsub/ndec,outfile[idm]);
        }
        printf("Processed %d DMs in %.2f s\n", ndm, (float)(clock() - startclock) / CLOCKS_PER_SEC);
    }

    // Close files
    for (i = 0; i < ndm; i++)
        fclose(outfile[i]);

    // Close files
    for (i = 0; i < 4; i++)
        fclose(rawfile[i]);

    // Free
    free(header);
    for (i = 0; i < 4; i++) {
        free(h5buf[i]);
        cudaFree(dh5buf);
        free(h5.rawfname[i]);
    }
    free(fbuf);
    free(dm);
    free(cbuf);
    free(outfile);

    cudaFree(dfbuf);
    cudaFree(dcbuf);
    cudaFree(cp1);
    cudaFree(cp2);
    cudaFree(cp1p);
    cudaFree(cp2p);
    cudaFree(dc);
    cudaFree(bs1);
    cudaFree(bs2);
    cudaFree(zavg);
    cudaFree(zstd);
    cudaFree(ddm);

    // Free plan
    cufftDestroy(ftc2cf);
    cufftDestroy(ftc2cb);

    return 0;
}
// !!!!!!
// This is a simple H5 reader for complex voltage data. Very little
// error checking is done.
struct header read_h5_header(char* fname)
{
    int i, len, ibeam, isap;
    struct header h;
    hid_t file_id, attr_id, sap_id, beam_id, memtype, group_id, space, coord_id;
    char* string, * pch;
    const char* stokes[] = { "_S0_","_S1_","_S2_","_S3_" };
    char* froot, * fpart, * ftest, group[32];
    FILE* file;

    // Find filenames
    for (i = 0; i < 4; i++) {
        pch = strstr(fname, stokes[i]);
        if (pch != NULL)
            break;
    }
    len = strlen(fname) - strlen(pch);
    froot = (char*)malloc(sizeof(char) * (len + 1));
    memset(froot, 0, sizeof(char) * (len + 1));
    fpart = (char*)malloc(sizeof(char) * (strlen(pch) - 6));
    memset(fpart, 0, sizeof(char) * (strlen(pch) - 6));
    ftest = (char*)malloc(sizeof(char) * (len + 20));
    memset(ftest,0, sizeof(char) * (len + 20));
    strncpy(froot, fname, len);
    strncpy(fpart, pch + 4, strlen(pch) - 7);

    // Check files
    for (i = 0; i < 4; i++) {
        // Format file name
        sprintf(ftest, "%s_S%d_%s.raw", froot, i, fpart);
        // Try to open
        if ((file = fopen(ftest, "r")) != NULL) {
            fclose(file);
        }
        else {
            fprintf(stderr, "Raw file %s not found\n", ftest);
            exit(-1);
        }
        h.rawfname[i] = (char*)malloc(sizeof(char) * (strlen(ftest) + 1));
        strcpy(h.rawfname[i], ftest);
    }

    // Get beam number
    for (i = 0; i < 4; i++) {
        pch = strstr(fname, "_B");
        if (pch != NULL)
            break;
    }
    sscanf(pch + 2, "%d", &ibeam);

    // Get SAP number
    for (i = 0; i < 4; i++) {
        pch = strstr(fname, "_SAP");
        if (pch != NULL)
            break;
    }
    sscanf(pch + 4, "%d", &isap);

    // Free
    free(froot);
    free(fpart);
    free(ftest);

    // Open file
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

    // Open subarray pointing group
    sprintf(group, "SUB_ARRAY_POINTING_%03d", isap);
    sap_id = H5Gopen(file_id, group, H5P_DEFAULT);

    // Start MJD
    attr_id = H5Aopen(sap_id, "EXPTIME_START_MJD", H5P_DEFAULT);
    H5Aread(attr_id, H5T_IEEE_F64LE, &h.tstart);
    H5Aclose(attr_id);

    // Declination
    attr_id = H5Aopen(sap_id, "POINT_DEC", H5P_DEFAULT);
    H5Aread(attr_id, H5T_IEEE_F64LE, &h.src_dej);
    H5Aclose(attr_id);

    // Right ascension
    attr_id = H5Aopen(sap_id, "POINT_RA", H5P_DEFAULT);
    H5Aread(attr_id, H5T_IEEE_F64LE, &h.src_raj);
    H5Aclose(attr_id);

    // Open beam
    sprintf(group, "BEAM_%03d", ibeam);
    beam_id = H5Gopen(sap_id, group, H5P_DEFAULT);

    // Number of samples
    attr_id = H5Aopen(beam_id, "NOF_SAMPLES", H5P_DEFAULT);
    H5Aread(attr_id, H5T_STD_U32LE, &h.nsamp);
    H5Aclose(attr_id);

    // Center frequency
    attr_id = H5Aopen(beam_id, "BEAM_FREQUENCY_CENTER", H5P_DEFAULT);
    H5Aread(attr_id, H5T_IEEE_F64LE, &h.fcen);
    H5Aclose(attr_id);

    // Center frequency unit
    attr_id = H5Aopen(beam_id, "BEAM_FREQUENCY_CENTER_UNIT", H5P_DEFAULT);
    memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(memtype, H5T_VARIABLE);
    H5Aread(attr_id, memtype, &string);
    H5Aclose(attr_id);
    if (strcmp(string, "Hz") == 0)
        h.fcen /= 1e6;

    // Channel bandwidth
    attr_id = H5Aopen(beam_id, "CHANNEL_WIDTH", H5P_DEFAULT);
    H5Aread(attr_id, H5T_IEEE_F64LE, &h.bwchan);
    H5Aclose(attr_id);

    // Center frequency unit
    attr_id = H5Aopen(beam_id, "CHANNEL_WIDTH_UNIT", H5P_DEFAULT);
    memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(memtype, H5T_VARIABLE);
    H5Aread(attr_id, memtype, &string);
    H5Aclose(attr_id);
    if (strcmp(string, "Hz") == 0)
        h.bwchan /= 1e6;

    // Get source
    attr_id = H5Aopen(beam_id, "TARGETS", H5P_DEFAULT);
    memtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(memtype, H5T_VARIABLE);
    H5Aread(attr_id, memtype, &string);
    H5Aclose(attr_id);
    strcpy(h.source_name, string);

    // Open coordinates
    coord_id = H5Gopen(beam_id, "COORDINATES", H5P_DEFAULT);

    // Open coordinate 0
    group_id = H5Gopen(coord_id, "COORDINATE_0", H5P_DEFAULT);

    // Sampling time
    attr_id = H5Aopen(group_id, "INCREMENT", H5P_DEFAULT);
    H5Aread(attr_id, H5T_IEEE_F64LE, &h.tsamp);
    H5Aclose(attr_id);

    // Close group
    H5Gclose(group_id);

    // Open coordinate 1
    group_id = H5Gopen(coord_id, "COORDINATE_1", H5P_DEFAULT);

    // Number of subbands
    attr_id = H5Aopen(group_id, "AXIS_VALUES_WORLD", H5P_DEFAULT);
    space = H5Aget_space(attr_id);
    h.nsub = H5Sget_simple_extent_npoints(space);
    H5Aclose(attr_id);

    // Close group
    H5Gclose(group_id);

    // Close coordinates
    H5Gclose(coord_id);

    // Close beam, sap and file
    H5Gclose(beam_id);
    H5Gclose(sap_id);
    H5Fclose(file_id);

    return h;
}

// Scale cufftComplex 
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s)
{
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Pointwise complex multiplication (and scaling)
static __global__ void PointwiseComplexMultiply(cufftComplex* a, cufftComplex* b, cufftComplex* c, int nx, int ny, int l, float scale)
{
    int i, j, k;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        k = i + nx * j;
        c[k] = ComplexScale(ComplexMul(a[k], b[i + nx * l]), scale);
    }
}

// Compute chirp
__global__ void compute_chirp(double fcen, double bw, float* dm, int nchan, int nbin, int nsub, int ndm, cufftComplex* c)
{
    int ibin, ichan, isub, idm, mbin, idx;
    double s, rt, t, f, fsub, fchan, bwchan, bwsub;

    // Number of channels per subband
    mbin = nbin / nchan;

    // Subband bandwidth
    bwsub = bw / nsub;

    // Channel bandwidth
    bwchan = bw / (nchan * nsub);

    // Indices of input data
    isub = blockIdx.x * blockDim.x + threadIdx.x;
    ichan = blockIdx.y * blockDim.y + threadIdx.y;
    idm = blockIdx.z * blockDim.z + threadIdx.z;

    // Keep in range
    if (isub < nsub && ichan < nchan && idm < ndm) {
        // Main constant
        s = 2.0 * M_PI * dm[idm] / DMCONSTANT;

        // Frequencies
        fsub = fcen - 0.5 * bw + bw * (float)isub / (float)nsub + 0.5 * bw / (float)nsub;
        fchan = fsub - 0.5 * bwsub + bwsub * (float)ichan / (float)nchan + 0.5 * bwsub / (float)nchan;

        // Loop over bins in channel
        for (ibin = 0; ibin < mbin; ibin++) {
            // Bin frequency
            f = -0.5 * bwchan + bwchan * (float)ibin / (float)mbin + 0.5 * bwchan / (float)mbin;

            // Phase delay
            rt = -f * f * s / ((fchan + f) * fchan * fchan);

            // Taper
            t = 1.0 / sqrt(1.0 + pow((f / (0.47 * bwchan)), 80));

            // Index
            idx = ibin + ichan * mbin + isub * mbin * nchan + idm * nsub * mbin * nchan;

            // Chirp
            c[idx].x = cos(rt) * t;
            c[idx].y = sin(rt) * t;
        }
    }

    return;
}

// Unpack the input buffer and generate complex timeseries. The output
// timeseries are padded with noverlap samples on either side for the
// convolution.
__global__ void unpack_and_padd(char* dbuf0, char* dbuf1, char* dbuf2, char* dbuf3, int nsamp, int nbin, int nfft, int nsub, int noverlap, cufftComplex* cp1, cufftComplex* cp2)
{
    int64_t ibin, ifft, isamp, isub, idx1, idx2;

    // Indices of input data
    ibin = blockIdx.x * blockDim.x + threadIdx.x;
    ifft = blockIdx.y * blockDim.y + threadIdx.y;
    isub = blockIdx.z * blockDim.z + threadIdx.z;

    // Only compute valid threads
    if (ibin < nbin && ifft < nfft && isub < nsub) {
        idx1 = ibin + nbin * isub + nsub * nbin * ifft;
        isamp = ibin + (nbin - 2 * noverlap) * ifft - noverlap;
        idx2 = isub + nsub * isamp;
        if (isamp < 0 || isamp >= nsamp) {
            cp1[idx1].x = 0.0;
            cp1[idx1].y = 0.0;
            cp2[idx1].x = 0.0;
            cp2[idx1].y = 0.0;
        }
        else {
            cp1[idx1].x = (float)dbuf0[idx2];
            cp1[idx1].y = (float)dbuf1[idx2];
            cp2[idx1].x = (float)dbuf2[idx2];
            cp2[idx1].y = (float)dbuf3[idx2];
        }
    }

    return;
}

// Since complex-to-complex FFTs put the center frequency at bin zero
// in the frequency domain, the two halves of the spectrum need to be
// swapped.
__global__ void swap_spectrum_halves(cufftComplex* cp1, cufftComplex* cp2, int nx, int ny)
{
    int64_t i, j, k, l, m;
    cufftComplex tp1, tp2;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx / 2 && j < ny) {
        if (i < nx / 2)
            k = i + nx / 2;
        else
            k = i - nx / 2;
        l = i + nx * j;
        m = k + nx * j;
        tp1.x = cp1[l].x;
        tp1.y = cp1[l].y;
        tp2.x = cp2[l].x;
        tp2.y = cp2[l].y;
        cp1[l].x = cp1[m].x;
        cp1[l].y = cp1[m].y;
        cp2[l].x = cp2[m].x;
        cp2[l].y = cp2[m].y;
        cp1[m].x = tp1.x;
        cp1[m].y = tp1.y;
        cp2[m].x = tp2.x;
        cp2[m].y = tp2.y;
    }

    return;
}

// After the segmented FFT the data is in a cube of nbin by nchan by
// nfft, where nbin and nfft are the time indices. Here we rearrange
// the 3D data cube into a 2D array of frequency and time, while also
// removing the overlap regions and detecting (generating Stokes I).
__global__ void transpose_unpadd_and_detect(cufftComplex* cp1, cufftComplex* cp2, int nbin, int nchan, int nfft, int nsub, int noverlap, int nsamp, float* fbuf)
{
    int64_t ibin, ichan, ifft, isub, isamp, idx1, idx2;

    ibin = blockIdx.x * blockDim.x + threadIdx.x;
    ichan = blockIdx.y * blockDim.y + threadIdx.y;
    ifft = blockIdx.z * blockDim.z + threadIdx.z;
    if (ibin < nbin && ichan < nchan && ifft < nfft) {
        // Loop over subbands
        for (isub = 0; isub < nsub; isub++) {
            // Padded array index
            //      idx1=ibin+nbin*isub+nsub*nbin*(ichan+nchan*ifft);
            idx1 = ibin + ichan * nbin + (nsub - isub - 1) * nbin * nchan + ifft * nbin * nchan * nsub;

            // Time index
            isamp = ibin + (nbin - 2 * noverlap) * ifft - noverlap;

            // Output array index
            idx2 = (nchan - ichan - 1) + isub * nchan + nsub * nchan * isamp;

            // Select data points from valid region
            if (ibin >= noverlap && ibin <= nbin - noverlap && isamp >= 0 && isamp < nsamp)
                fbuf[idx2] = cp1[idx1].x * cp1[idx1].x + cp1[idx1].y * cp1[idx1].y + cp2[idx1].x * cp2[idx1].x + cp2[idx1].y * cp2[idx1].y;
        }
    }

    return;
}

void send_string(const char* string, FILE* file)
{
    int len;

    len = strlen(string);
    fwrite(&len, sizeof(int), 1, file);
    fwrite(string, sizeof(char), len, file);

    return;
}

void send_float(const char* string, float x, FILE* file)
{
    send_string(string, file);
    fwrite(&x, sizeof(float), 1, file);

    return;
}

void send_int(const char* string, int x, FILE* file)
{
    send_string(string, file);
    fwrite(&x, sizeof(int), 1, file);

    return;
}

void send_double(const char* string, double x, FILE* file)
{
    send_string(string, file);
    fwrite(&x, sizeof(double), 1, file);

    return;
}

double dec2sex(double x)
{
    double d, sec, min, deg;
    char sign;
    char tmp[32];

    sign = (x < 0 ? '-' : ' ');
    x = 3600.0 * fabs(x);

    sec = fmod(x, 60.0);
    x = (x - sec) / 60.0;
    min = fmod(x, 60.0);
    x = (x - min) / 60.0;
    deg = x;

    sprintf(tmp, "%c%02d%02d%09.6lf", sign, (int)deg, (int)min, sec);
    sscanf(tmp, "%lf", &d);

    return d;
}

void write_filterbank_header(struct header h, FILE* file)
{
    double ra, de;


    ra = dec2sex(h.src_raj / 15.0);
    de = dec2sex(h.src_dej);

    send_string("HEADER_START", file);
    send_string("rawdatafile", file);
    send_string(h.rawfname[0], file);
    send_string("source_name", file);
    send_string(h.source_name, file);
    send_int("machine_id", 11, file);
    send_int("telescope_id", 11, file);
    send_double("src_raj", ra, file);
    send_double("src_dej", de, file);
    send_int("data_type", 1, file);
    send_double("fch1", h.fch1, file);
    send_double("foff", h.foff, file);
    send_int("nchans", h.nchan, file);
    send_int("nbeams", 0, file);
    send_int("ibeam", 0, file);
    send_int("nbits", h.nbit, file);
    send_double("tstart", h.tstart, file);
    send_double("tsamp", h.tsamp, file);
    send_int("nifs", 1, file);
    send_string("HEADER_END", file);

    return;
}

// Compute segmented sums for later computation of offset and scale
__global__ void compute_block_sums(float* z, int nchan, int nblock, int nsum, float* bs1, float* bs2)
{
    int64_t ichan, iblock, isum, idx1, idx2;

    ichan = blockIdx.x * blockDim.x + threadIdx.x;
    iblock = blockIdx.y * blockDim.y + threadIdx.y;
    if (ichan < nchan && iblock < nblock) {
        idx1 = ichan + nchan * iblock;
        bs1[idx1] = 0.0;
        bs2[idx1] = 0.0;
        for (isum = 0; isum < nsum; isum++) {
            idx2 = ichan + nchan * (isum + iblock * nsum);
            bs1[idx1] += z[idx2];
            bs2[idx1] += z[idx2] * z[idx2];
        }
    }

    return;
}

// Compute segmented sums for later computation of offset and scale
__global__ void compute_channel_statistics(int nchan, int nblock, int nsum, float* bs1, float* bs2, float* zavg, float* zstd)
{
    int64_t ichan, iblock, idx1;
    double s1, s2;

    ichan = blockIdx.x * blockDim.x + threadIdx.x;
    if (ichan < nchan) {
        s1 = 0.0;
        s2 = 0.0;
        for (iblock = 0; iblock < nblock; iblock++) {
            idx1 = ichan + nchan * iblock;
            s1 += bs1[idx1];
            s2 += bs2[idx1];
        }
        zavg[ichan] = s1 / (float)(nblock * nsum);
        zstd[ichan] = s2 / (float)(nblock * nsum) - zavg[ichan] * zavg[ichan];
        zstd[ichan] = sqrt(zstd[ichan]);
    }

    return;
}

// Redigitize the filterbank to 8 bits in segments
__global__ void redigitize(float* z, int nchan, int nblock, int nsum, float* zavg, float* zstd, float zmin, float zmax, unsigned char* cz)
{
    int64_t ichan, iblock, isum, idx1;
    float zoffset, zscale;

    ichan = blockIdx.x * blockDim.x + threadIdx.x;
    iblock = blockIdx.y * blockDim.y + threadIdx.y;
    if (ichan < nchan && iblock < nblock) {
        zoffset = zavg[ichan] - zmin * zstd[ichan];
        zscale = (zmin + zmax) * zstd[ichan];

        for (isum = 0; isum < nsum; isum++) {
            idx1 = ichan + nchan * (isum + iblock * nsum);
            z[idx1] -= zoffset;
            z[idx1] *= 256.0 / zscale;
            cz[idx1] = (unsigned char)z[idx1];
            if (z[idx1] < 0.0) cz[idx1] = 0;
            if (z[idx1] > 255.0) cz[idx1] = 255;
        }
    }

    return;
}

// Decimate and Redigitize the filterbank to 8 bits in segments
__global__ void decimate_and_redigitize(float* z, int ndec, int nchan, int nblock, int nsum, float* zavg, float* zstd, float zmin, float zmax, unsigned char* cz)
{
    int64_t ichan, iblock, isum, idx1, idx2, idec;
    float zoffset, zscale, ztmp;

    ichan = blockIdx.x * blockDim.x + threadIdx.x;
    iblock = blockIdx.y * blockDim.y + threadIdx.y;
    if (ichan < nchan && iblock < nblock) {
        zoffset = zavg[ichan] - zmin * zstd[ichan];
        zscale = (zmin + zmax) * zstd[ichan];

        for (isum = 0; isum < nsum; isum += ndec) {
            idx2 = ichan + nchan * (isum / ndec + iblock * nsum / ndec);
            for (idec = 0, ztmp = 0.0; idec < ndec; idec++) {
                idx1 = ichan + nchan * (isum + idec + iblock * nsum);
                ztmp += z[idx1];
            }
            ztmp /= (float)ndec;
            ztmp -= zoffset;
            ztmp *= 256.0 / zscale;
            cz[idx2] = (unsigned char)ztmp;
            if (ztmp < 0.0) cz[idx2] = 0;
            if (ztmp > 255.0) cz[idx2] = 255;
        }
    }

    return;
}

//-----------------------------------------------------------------
__global__ void kernel_Sum_cols(unsigned char* d_inp, const int NRows, const int NCols, int* d_arrOut)
{
    extern __shared__ int sdata[];

    unsigned char* pTemp = d_inp + NCols * blockIdx.x + threadIdx.x;

    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    if (tid >= NCols)
    {
        sdata[tid] = 0;

    }
    else
    {
        int localSum = 0;



        while (i < NCols)
        {
            localSum += (int)(pTemp[i]);
            pTemp += blockDim.x;
            i += blockDim.x;

        }


    }
    __syncthreads();

    // Parallel reduction within the block to sum partial sums
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }

    // Only thread 0 within each block computes the block's sum
    if (tid == 0)
    {
        d_arrOut[blockIdx.x] = sdata[0];

    }
    __syncthreads();

}
//---------------------------------------------------
int readHeader5InBinary(const char* strFilePath, header& Head)
{

    // Open file for reading in binary mode
    FILE* file = fopen(strFilePath, "rb");

    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }



    // Write the variables to the file

    fread(&Head.headersize, sizeof(int64_t), 1, file);
    fread(&Head.buffersize, sizeof(int64_t), 1, file);//
    fread(&Head.nchan, sizeof(unsigned int), 1, file);
    fread(&Head.nsamp, sizeof(unsigned int), 1, file);
    fread(&Head.nbit, sizeof(unsigned int), 1, file);
    fread(&Head.nif, sizeof(unsigned int), 1, file);
    fread(&Head.nsub, sizeof(unsigned int), 1, file);


    fread(&Head.machine_id, sizeof(int), 1, file);
    fread(&Head.telescope_id, sizeof(int), 1, file);
    fread(&Head.nbeam, sizeof(int), 1, file);
    fread(&Head.ibeam, sizeof(int), 1, file);
    fread(&Head.sumif, sizeof(int), 1, file);


    fread(&Head.tstart, sizeof(double), 1, file);
    fread(&Head.tsamp, sizeof(double), 1, file);
    fread(&Head.fch1, sizeof(double), 1, file);
    fread(&Head.foff, sizeof(double), 1, file);
    fread(&Head.fcen, sizeof(double), 1, file);
    fread(&Head.bwchan, sizeof(double), 1, file);


    fread(&Head.src_raj, sizeof(double), 1, file);
    fread(&Head.src_dej, sizeof(double), 1, file);
    fread(&Head.az_start, sizeof(double), 1, file);
    fread(&Head.za_start, sizeof(double), 1, file);




    fread(Head.source_name, sizeof(char), 80, file);
    fread(Head.ifstream, sizeof(char), 8, file);
    fread(Head.inpfile, sizeof(char), 8, file);


    for (int i = 0; i < 4; ++i)
    {
        int len = 0;
        fread(&len, sizeof(int), 1, file);
        Head.rawfname[i] = new char[len + 1];
        memset(Head.rawfname[i], 0, (1 + len) * sizeof(char));

        fread(Head.rawfname[i], sizeof(char), len, file);

    }

    // Close the file
    fclose(file);

    return 0;
}
//--------------------------------------------------
int writeOptions(const char* strFilePath, const int noverlap, const int nbin, const int ndec, const char* obsid
    , const int part, const int device)
{
    FILE* file = fopen(strFilePath, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    // Write the variables to the file
    fwrite(&noverlap, sizeof(int), 1, file);
    fwrite(&nbin, sizeof(int), 1, file);//
    fwrite(&ndec, sizeof(int), 1, file);//
    int len = strlen(obsid);
    fwrite(&len, sizeof(int), 1, file);
    fwrite(obsid, sizeof(char), len, file);
    fwrite(&part, sizeof(int), 1, file);
    fwrite(&device, sizeof(int), 1, file);
    fclose(file);
    return 0;

}
//-------------------------------------

int readOptions(const char* strFilePath, int& noverlap, int& nbin, int& ndec, char* obsid
    , int& part, int& device)
{
    // Open file for reading in binary mode
    FILE* file = fopen(strFilePath, "rb");

    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    // Write the variables to the file
    fread(&noverlap, sizeof(int), 1, file);
    fread(&nbin, sizeof(int), 1, file);//
    //fread(&ndec, sizeof(int), 1, file);//
    ndec = 4;
    int len = 0;
    fread(&len, sizeof(int), 1, file);
    fread(obsid, sizeof(char), len, file);
    fread(&part, sizeof(int), 1, file);
    fread(&device, sizeof(int), 1, file);


    // Close the file
    fclose(file);

    return 0;
}