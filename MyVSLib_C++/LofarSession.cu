#include "LofarSession.cuh"
#include <string>
#include "stdio.h"
#include <iostream>
#include "Chunk.cuh"
#include "OutChunk.h"
#include <cufft.h>
#include <complex>
#include <vector>
#include "yr_cart.h"

#include <stdlib.h>
#include "Fragment.h"
#include "helper_functions.h"
#include "helper_cuda.h"
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
CLofarSession::~CLofarSession()
{ 
    /*for (int i = 0; i < 4; ++i)
    {
        if (m_rawfile[i])
        {
            fclose(m_rawfile[i]);
        }        
       
    }
    
    if (m_wb_file)
    {
        fclose(m_wb_file);
    }
    m_wb_file = NULL;*/

    if (m_pvctSuccessHeaders)
    {
        delete m_pvctSuccessHeaders;
    }

}
//-------------------------------------------
CLofarSession::CLofarSession()
{   
    /*for (int i = 0; i < 4; ++i)
    {
        m_rawfile[i] = NULL;
        
    }*/
    memset(m_strInputPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_header = CTelescopeHeader();
    m_pulse_length = 1.0E-6;
    m_d_max = 0.;
    m_sigma_bound = 10.;
    m_length_sum_wnd = 10;
}

//--------------------------------------------
CLofarSession::CLofarSession(const  CLofarSession& R)
{        
    memcpy(m_strInputPath, R.m_strInputPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
    m_header = R.m_header;  
    m_pulse_length = R.m_pulse_length;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
}

//-------------------------------------------
CLofarSession& CLofarSession::operator=(const CLofarSession& R)
{
    if (this == &R)
    {
        return *this;
    }
    
    /*for (int i = 0; i < 4; ++i)
    {
        m_rawfile[i] = R.m_rawfile[i];
    }*/
    memcpy(m_strInputPath, R.m_strInputPath, MAX_PATH_LENGTH * sizeof(char));
    memcpy(m_strOutPutPath, R.m_strOutPutPath, MAX_PATH_LENGTH * sizeof(char));
    if (m_pvctSuccessHeaders)
    {
        m_pvctSuccessHeaders = R.m_pvctSuccessHeaders;
    }
    m_header = R.m_header;    
    m_pulse_length = R.m_pulse_length;
    m_d_max = R.m_d_max;
    m_sigma_bound = R.m_sigma_bound;
    m_length_sum_wnd = R.m_length_sum_wnd;
    return *this;
}

//--------------------------------- 
CLofarSession::CLofarSession(const char* strInputPath, const char* strOutPutPath, const float pulse_length
,const double d_max, const float sigma_bound, const int length_sum_wnd)
{
    strcpy(m_strOutPutPath, strOutPutPath);
    strcpy(m_strInputPath, strInputPath);
    /*m_rbFile = fopen(strInputPath, "rb");    
    m_wb_file = fopen(strOutPutPath, "wb");*/
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_pulse_length = pulse_length;
    m_d_max = d_max;
    m_sigma_bound = sigma_bound;
    m_length_sum_wnd = length_sum_wnd;
}

//
//---------------------------------------------------------------
int CLofarSession::launch()
{
    // 1. reading hdf5 header:
    header_h5 Hdr5 = read_h5_header(m_strInputPath);
    //!1

    //2. open 4 raw files to read
    FILE* rawfile[4];
    for (int i = 0; i < 4; ++i)
    {
        if (!(rawfile[i] = fopen(Hdr5.rawfname[i], "rb")))
        {
            printf("Can't open raw file number %d", i);
            return -1;
        }
    }
    // 2!   

    // calc quantity LofarSessions
    unsigned long long ilength = 0;

   



    // 2.allocation memory for parametrs in CPU
    int nbits = 8;
    //float chanBW = Hdr5.bwchan;
    int npol = 4;
    bool bdirectIO = 0;
    //float centfreq = Hdr5.fcen;
    //int nchan = Hdr5.nsub;
   // float obsBW = nchan * chanBW;
   //long long nsamples = Hdr5.nsamp;
    EN_telescope TELESCOP = LOFAR;
    int ireturn = 0;
    // !2



    if (nbits / 8 != sizeof(inp_type_))
    {
        std::cout << "check up Constants.h, inp_type_  " << std::endl;
        return -1;
    }

    m_header = CTelescopeHeader(
        nbits
        , Hdr5.bwchan
        , npol
        , bdirectIO
        , Hdr5.fcen
        , Hdr5.nsub
        , Hdr5.nsub * Hdr5.bwchan
        , Hdr5.nsamp
        , TELESCOP
        , Hdr5.tsamp * 1.E-6
    );
    cufftHandle plan0 = NULL;
    cufftHandle plan1 = NULL;
    int lenChunk = 0;
    inp_type_ * d_parrInput = NULL;
    cufftComplex* pcmparrRawSignalCur = NULL;
    CFdmtU fdmt;
    void* pAuxBuff_fdmt = NULL;
    fdmt_type_* d_arrfdmt_norm = NULL;
    cufftComplex* pcarrTemp = NULL; //2	
    cufftComplex* pcarrCD_Out = NULL;//3
    cufftComplex* pcarrBuff = NULL;//3
    char* pInpOutBuffFdmt = NULL;
    CChunk* pChunk = new CChunk();
    

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately
    }

    CChunk::preparations_and_memoryAllocations(m_header
        , m_pulse_length
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd
        , &lenChunk
        , &plan0, &plan1, &fdmt, (char**) & d_parrInput, &pcmparrRawSignalCur
        , &pAuxBuff_fdmt, &d_arrfdmt_norm, &pcarrTemp, &pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt,  &pChunk);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error appropriately
    }

    const long long QUantDownloadingBytesForEachFile = lenChunk * m_header.m_nchan / 8 * m_header.m_nbits;
    const long long QUantBlockComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;




    

    // quantity loop iterations (blocks)
    const int IChank = (m_header.m_nblocksize + lenChunk - 1) / lenChunk;
    std::cout << "!!!  Quantity of Chuncks = " << IChank <<"  !!!"<< std::endl;
    for (int nC = 0; nC < IChank; ++nC)
    {

        cout << "                               Chunk =  " << nC << endl;
        // 3.1. 
        if (nC == IChank - 1)
        {
            for (int i = 0; i < 4; ++i)
            {
                fseek(rawfile[i], lenChunk, SEEK_END);
            }
        }

        for (int i = 0; i < 4; i++)
        {
            int nread = fread(&(((char*)d_parrInput)[QUantDownloadingBytesForEachFile * i]), sizeof(char), QUantDownloadingBytesForEachFile, rawfile[i]);
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
        const dim3 gridSize((lenChunk + block_size.x - 1) / block_size.x,1, 1);
        unpackInput << < gridSize, block_size >> > (pcmparrRawSignalCur, d_parrInput, lenChunk, m_header.m_nchan, m_header.m_npol);

         std::vector<std::complex<float>> data2(lenChunk, 0);
    cudaMemcpy(data2.data(), pcmparrRawSignalCur, lenChunk * sizeof(std::complex<float>),
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
            // Handle the error appropriately
        }

        pChunk->set_chunkid(nC);
        pChunk->fncChunkProcessing_gpu(pcmparrRawSignalCur
            , pAuxBuff_fdmt
            , pcarrTemp
            , pcarrCD_Out
            , pcarrBuff
            , pInpOutBuffFdmt
            , d_arrfdmt_norm
            , plan0
            , plan1
            , m_pvctSuccessHeaders);

    }
    cudaFree(pcmparrRawSignalCur);
    cudaFree(d_arrfdmt_norm);
    cudaFree(pAuxBuff_fdmt);
    cudaFree(pcarrTemp); //2	
    cudaFree(pcarrCD_Out);//3
    cudaFree(pcarrBuff);//3
    cudaFree(pInpOutBuffFdmt);
    cudaFree(d_parrInput);
    delete pChunk;    
    cufftDestroy(plan0);
    cufftDestroy(plan1);
    return 0;
}

//---------------------------------------------------------
bool CLofarSession::analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg)
{
    // 1. reading hdf5 header:
    header_h5 Hdr5 = read_h5_header(m_strInputPath);
    //!1


    //2. open 4 raw files to read
    FILE* rawfile[4];
    for (int i = 0; i < 4; ++i)
    {
        if (!(rawfile[i] = fopen(Hdr5.rawfname[i], "rb")))
        {
            printf("Can't open raw file number %d", i);
            return -1;
        }

    }
    // 2!

    // retrieve parameters from Hdr5:

    // calc quantity LofarSessions
    unsigned long long ilength = 0;

    // 1. blocks quantity calculation



    // 2.allocation memory for parametrs in CPU
    int nbits = 8;
    //float chanBW = Hdr5.bwchan;
    int npol = 4;
    bool bdirectIO = 0;
    //float centfreq = Hdr5.fcen;
    //int nchan = Hdr5.nsub;
   // float obsBW = nchan * chanBW;
   //long long nsamples = Hdr5.nsamp;
    EN_telescope TELESCOP = LOFAR;
    int ireturn = 0;
    // !2



    if (nbits / 8 != sizeof(inp_type_))
    {
        std::cout << "check up Constants.h, inp_type_  " << std::endl;
        return -1;
    }

    m_header = CTelescopeHeader(
        nbits
        , Hdr5.bwchan
        , npol
        , bdirectIO
        , Hdr5.fcen
        , Hdr5.nsub
        , Hdr5.nsub * Hdr5.bwchan
        , Hdr5.nsamp
        , TELESCOP
        , Hdr5.tsamp * 1.E-6
    );
    cufftHandle plan0 = NULL;
    cufftHandle plan1 = NULL;
    int lenChunk = 0;
    char* d_parrInput = NULL;
    cufftComplex* pcmparrRawSignalCur = NULL;
    CFdmtU fdmt;
    void* pAuxBuff_fdmt = NULL;
    fdmt_type_* d_arrfdmt_norm = NULL;
    cufftComplex* pcarrTemp = NULL; //2	
    cufftComplex* pcarrCD_Out = NULL;//3
    cufftComplex* pcarrBuff = NULL;//3
    char* pInpOutBuffFdmt = NULL;
    CChunk* pChunk = new CChunk();
    
    CChunk::preparations_and_memoryAllocations(m_header
        , m_pulse_length
        , m_d_max
        , m_sigma_bound
        , m_length_sum_wnd
        , &lenChunk
        , &plan0, &plan1, &fdmt,(char**) & d_parrInput, &pcmparrRawSignalCur
        , &pAuxBuff_fdmt, &d_arrfdmt_norm, &pcarrTemp, &pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt,  &pChunk);

    

    const long long QUantDownloadingBytesForEachFile = lenChunk * m_header.m_nchan / 8 * m_header.m_nbits;
    const long long QUantTotalBytesForChunk = QUantDownloadingBytesForEachFile * m_header.m_npol;
    const long long QUantBlockComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;

    // 3. Performing a loop using the variable nS, nS = 0,..,IChank. 
    //IChank - number of bulks
     // quantity loop iterations (blocks)
    const int IChank = (m_header.m_nblocksize + QUantTotalBytesForChunk - 1) / QUantTotalBytesForChunk;
    for (int nC = 0; nC < IChank; ++nC)
    {

        cout << "                               BLOCK=  " << nC << endl;
        // 3.1. 
        if (nC == IChank - 1)
        {
            for (int i = 0; i < 4; ++i)
            {
                fseek(rawfile[i], lenChunk, SEEK_END);
            }
        }

        for (int i = 0; i < 4; i++)
        {
            int nread = fread(&(((char*)d_parrInput)[QUantDownloadingBytesForEachFile]), sizeof(char), QUantDownloadingBytesForEachFile, rawfile[i]);
            if (nread == 0)
                break;

        }
        cudaDeviceSynchronize();

        if (nC != outChunkHeader.m_numChunk - 1)
        {
            continue;
        }
        if (nC >= outChunkHeader.m_numChunk)
        {
            break;
        }

        //
        const dim3 block_size(32, 1024, 1);
        const dim3 gridSize((m_header.m_nchan + block_size.x - 1) / block_size.x, (lenChunk + block_size.y - 1) / block_size.y, 1);
        unpackInput << < gridSize, block_size >> > (pcmparrRawSignalCur, (inp_type_ * )d_parrInput, lenChunk, m_header.m_nchan, npol);

        if (!pChunk->detailedChunkProcessing(outChunkHeader
            , plan0
            , plan1
            , pcmparrRawSignalCur
            , d_arrfdmt_norm
            , pAuxBuff_fdmt
            , pcarrTemp
            , pcarrCD_Out
            , pcarrBuff
            , pInpOutBuffFdmt
            , pFRg))
        {
            cudaFree(pcmparrRawSignalCur);
            cudaFree(d_arrfdmt_norm);
            cudaFree(pAuxBuff_fdmt);
            cudaFree(pcarrTemp); //2	
            cudaFree(pcarrCD_Out);//3
            cudaFree(pcarrBuff);//3
            cudaFree(pInpOutBuffFdmt);
            cudaFree(d_parrInput);
            delete pChunk;
            
            cufftDestroy(plan0);
            cufftDestroy(plan1);
            return false;
        }
        else
        {
            break;
        }

    }
    cudaFree(pcmparrRawSignalCur);
    cudaFree(d_arrfdmt_norm);
    cudaFree(pAuxBuff_fdmt);
    cudaFree(pcarrTemp); //2	
    cudaFree(pcarrCD_Out);//3
    cudaFree(pcarrBuff);//3
    cudaFree(pInpOutBuffFdmt);
    cudaFree(d_parrInput);
    delete pChunk;
    
    cufftDestroy(plan0);
    cufftDestroy(plan1);
    return true;
}


////-----------------------------------------------------------------
//
//long long CLofarSession::calcLenChunk(const int nsft)
//{
//    const int nchan_actual = nsft * m_header.m_nchan;
//
//    long long len = 0;
//    for (len = 1 << 9; len < 1 << 30; len <<= 1)
//    {
//        CFdmtU fdmt(
//            m_header.m_centfreq - m_header.m_chanBW * m_header.m_nchan / 2.
//            , m_header.m_centfreq + m_header.m_chanBW * m_header.m_nchan / 2.
//            , nchan_actual
//            , len
//            , m_pulse_length
//            , m_d_max
//            , nsft
//        );
//        long long size0 = fdmt.calcSizeAuxBuff_fdmt_();
//        long long size_fdmt_inp = fdmt.calc_size_input();
//        long long size_fdmt_out = fdmt.calc_size_output();
//        long long size_fdmt_norm = size_fdmt_out;
//        long long irest = m_header.m_nchan * m_header.m_npol * m_header.m_nbits / 8 // input buff
//            + m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex)
//            + 3 * m_header.m_nchan * m_header.m_npol * sizeof(cufftComplex) / 2
//            + 2 * m_header.m_nchan * sizeof(float);
//        irest *= len;
//
//        long long rez = size0 + size_fdmt_inp + size_fdmt_out + size_fdmt_norm + irest;
//        if (rez > 0.95 * TOtal_GPU_Bytes)
//        {
//            return len / 2;
//        }
//
//    }
//    return -1;
//
//}


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
//-------------------------------------------------------


//INPUT:
__global__
void unpackInput(cufftComplex* pcmparrRawSignalCur, inp_type_* d_parrInput, const int  lenChunk
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
            =  (float)d_parrInput[(1 + j * npol / 2) * colsInp + numColInp0 + i];
        }
    }
    
}




