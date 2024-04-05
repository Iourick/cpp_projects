#include "Session_gpu_lofar.cuh"
#include <string>
#include "stdio.h"
#include <iostream>

#include "OutChunk.h"
#include <cufft.h>
#include "FdmtU.cuh"
#include <stdlib.h>
#include "Fragment.cuh"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "yr_cart.h"
#include "Chunk.cuh"
#include <complex>
#include "Session_gpu.cuh"

#include <hdf5.h> 
cudaError_t cudaStatus;
struct header_h5 {
    int64_t headersize, buffersize;
    unsigned int nchan, nsamp, nbit, nif, nsub;
    int machine_id, telescope_id, nbeam, ibeam, sumif;
    double tstart, tsamp, fch1, foff, fcen, bwchan;
    double src_raj, src_dej, az_start, za_start;
    char source_name[80] = { 0 }, ifstream[8] = { 0 }, inpfile[80] = { 0 };
    char rawfname[4][300] = { 0 };
};
//-------------------------------------------
CSession_gpu_lofar::CSession_gpu_lofar() :CSession_gpu()
{
}

//--------------------------------------------
CSession_gpu_lofar::CSession_gpu_lofar(const  CSession_gpu_lofar& R) :CSession_gpu(R)
{
}

//-------------------------------------------
CSession_gpu_lofar& CSession_gpu_lofar::operator=(const CSession_gpu_lofar& R)
{
    if (this == &R)
    {
        return *this;
    }
    CSession_gpu:: operator= (R);

    return *this;
}

//--------------------------------- 
CSession_gpu_lofar::CSession_gpu_lofar(const char* strlofarPath, const char* strOutPutPath, const float t_p
    , const double d_max, const float sigma_bound, const int length_sum_wnd)
    :CSession_gpu(strlofarPath, strOutPutPath, t_p, d_max, sigma_bound, length_sum_wnd)
{
}
//------------------------------------
int CSession_gpu_lofar::calcQuantBlocks( unsigned long long* pilength)
{    
    return 1;
}


//----------------------------------------------------
bool CSession_gpu_lofar::openFileReadingStream(FILE** &prb_File)
{
    // 1. reading hdf5 header:
    header_h5 Hdr5 = read_h5_header(m_strInpPath);
    //!1

    prb_File = (FILE**)realloc(prb_File, 4 * sizeof(FILE*));
    if (!prb_File) {
        // Handle memory reallocation failure
        return false;
    }

    //2. open 4 raw files to read
    FILE* rawfile[4];
    for (int i = 0; i < 4; ++i)
    {
        if (!(rawfile[i] = fopen(Hdr5.rawfname[i], "rb")))
        {
            printf("Can't open raw file number %d", i);
            return false;
        }
        prb_File[i] = rawfile[i];
    }
    return true;
}
//---------------------------------------------------------------
//-----------------------------------------------------------------
struct header_h5 read_h5_header(char* fname)
{
    int i, len, ibeam, isap;
    struct header_h5 h;
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
    memset(ftest, 0, sizeof(char) * (len + 20));
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
//---------------------------------------------
bool CSession_gpu_lofar::readTelescopeHeader(FILE* r_file
    , int* nbits
    , float* chanBW
    , int* npol
    , bool* bdirectIO
    , float* centfreq
    , int* nchan
    , float* obsBW
    , long long* nblocksize
    , EN_telescope* TELESCOP
    , float* tresolution
)
{
   
    header_h5 Hdr5 = read_h5_header(m_strInpPath);
   
    *nbits = 8;
    *chanBW = Hdr5.bwchan;
    *npol = 4;
    *bdirectIO = 0;
    *centfreq = Hdr5.fcen;
    *nchan = Hdr5.nsub;
    *obsBW = (*nchan) * (*chanBW) ;
    *nblocksize = Hdr5.nsamp * (*npol) * Hdr5.nsub* (*nbits)/8;
    *TELESCOP = LOFAR;
    *tresolution = Hdr5.tsamp;
   
    return true;
}
//----------------------------------------------------------------------
bool  CSession_gpu_lofar::createCurrentTelescopeHeader(FILE** prb_File)
{
    // there is nothing to do
    return true;
}
//-----------------------------------------------------------
//----------------------------------------------------------
void CSession_gpu_lofar::download_and_unpack_chunk(FILE** prb_File, const long long lenChunk, const int j
    , inp_type_* d_parrInput, cufftComplex* pcmparrRawSignalCur)
{

    const long long QUantDownloadingBytesForEachFile = lenChunk * m_header.m_nchan / 8 * m_header.m_nbits;
    const long long QUantTotalBytesForChunk = QUantDownloadingBytesForEachFile * m_header.m_npol;
    const long long QUantBlockComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;

    
    const int IChank = (m_header.m_nblocksize + QUantTotalBytesForChunk - 1) / QUantTotalBytesForChunk;
    if (j == IChank - 1)
    {
        for (int i = 0; i < 4; ++i)
        {
            fseek(prb_File[i], lenChunk, SEEK_END);
        }
    }

    for (int i = 0; i < 4; i++)
    {
        int nread = fread(&(((char*)d_parrInput)[QUantDownloadingBytesForEachFile * i]), sizeof(char), QUantDownloadingBytesForEachFile, prb_File[i]);
        if (nread == 0)
            break;

    }
    cudaDeviceSynchronize();    

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately
    }
    //
    const dim3 block_size(1024, 1, 1);
    const dim3 gridSize((lenChunk + block_size.x - 1) / block_size.x, 1, 1);
    
    unpackInput_L << < gridSize, block_size >> > (pcmparrRawSignalCur, (inp_type_*)d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);

}

//INPUT:
__global__
void unpackInput_L(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
    , const int  NChan, const int  npol)
{
    const int itime = threadIdx.x + blockDim.x * blockIdx.x; // down channels, number of channel
    if (itime >= lenChunk)
    {
        return;
    }
    int numColInp0 = itime * NChan;

    int colsInp = lenChunk * NChan;


    for (int i = 0; i < NChan; ++i)
    {
        for (int j = 0; j < npol / 2; ++j)
        {
            pcmparrRawSignalCur[(i * npol / 2 + j) * lenChunk + itime].x
                = (float)d_parrInput[j * npol / 2 * colsInp + numColInp0 + i];
            pcmparrRawSignalCur[(i * npol / 2 + j) * lenChunk + itime].y
                = (float)d_parrInput[(1 + j * npol / 2) * colsInp + numColInp0 + i];
        }
    }

}
//---------------------------------------------------------------------
void CSession_gpu_lofar::rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes)
{

}
//---------------------------------------------------------------------------------------------------
//----------------------------------------------------------------
bool CSession_gpu_lofar::closeFileReadingStream(FILE**& prb_File)
{
    for (int i = 0; i < 4; ++i) {
        if (prb_File[i]) {
            fclose(prb_File[i]);
        }
    }

    return true;
}