#include "SessionB.h"
#include <string>
#include "stdio.h"
#include <iostream>

#include <vector>
#include "OutChunk.h"

#include "CFdmtC.h"
#include <stdlib.h>
#include <fftw3.h>

//#include "Fragment.h"

//#include "yr_cart.h"

//#include "Chunk_cpu.h"
#include <complex>
 

CSessionB::~CSessionB()
{ 
    if (m_pvctSuccessHeaders)
    {
        delete m_pvctSuccessHeaders;
    }

}
//-------------------------------------------
CSessionB::CSessionB()
{     
    memset( m_strInpPath, 0, MAX_PATH_LENGTH * sizeof(char));
    memset(m_strOutPutPath, 0, MAX_PATH_LENGTH * sizeof(char));
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_header = CTelescopeHeader();
    m_pulse_length = 1.0E-6;
    m_d_max = 0.;
    m_sigma_bound = 10.;
    m_length_sum_wnd = 10;
    //LenChunk = 0;
    m_nbin = 0;
    m_nfft = 0;
}

//--------------------------------------------
CSessionB::CSessionB(const  CSessionB& R)
{  
    memcpy( m_strInpPath, R. m_strInpPath, MAX_PATH_LENGTH * sizeof(char));
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
    m_nbin = R.m_nbin;
    m_nfft = R.m_nfft;
}

//-------------------------------------------
CSessionB& CSessionB::operator=(const CSessionB& R)
{
    if (this == &R)
    {
        return *this;
    }     
    memcpy( m_strInpPath, R. m_strInpPath, MAX_PATH_LENGTH * sizeof(char));
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
   //LenChunk = R.LenChunk;
    m_nbin = R.m_nbin;
    m_nfft = R.m_nfft;

    return *this;
}

//--------------------------------- 
CSessionB::CSessionB(const char* strGuppiPath, const char* strOutPutPath, const float t_p
    , const double d_min, const double d_max, const float sigma_bound, const int length_sum_wnd, const int nbin, const int nfft)
{
    strcpy(m_strOutPutPath, strOutPutPath);
    strcpy( m_strInpPath, strGuppiPath);   
    m_pvctSuccessHeaders = new std::vector<COutChunkHeader>();
    m_pulse_length = t_p;
    m_d_min = d_min;
    m_d_max = d_max;
    m_sigma_bound = sigma_bound;
    m_length_sum_wnd = length_sum_wnd;
    m_nbin = nbin;
    m_nfft = nfft;
}
//------------------------------------
int CSessionB::calcQuantBlocks(unsigned long long* pilength)
{
    return -1;
}

//---------------------------------------------------------------
int CSessionB::launch(std::vector<std::vector<float>>* pvecImg, int *pmsamp)
{
    // calc quantity sessions
    unsigned long long ilength = 0;
    // 1. blocks quantity calculation  
    const int IBlock = calcQuantBlocks(&ilength);  
   
    //!1 
    
    FILE* rb_File = fopen(m_strInpPath, "rb");
    if (!readTelescopeHeader(
        rb_File
        , &m_header.m_nbits
        , &m_header.m_chanBW
        , &m_header.m_npol
        , &m_header.m_bdirectIO
        , &m_header.m_centfreq
        , &m_header.m_nchan
        , &m_header.m_obsBW
        , &m_header.m_nblocksize
        , &m_header.m_TELESCOP
        , &m_header.m_tresolution
    )
        )
    {
        return false;
    }
    fclose(rb_File);
    if (m_header.m_nbits / 8 != sizeof(inp_type_))
    {
        std::cout << "check up Constants.h, inp_type_  " << std::endl;
        return -1;
    }  

    // Calculate the optimal channels per subband
    const int n_p = int(ceil(m_pulse_length / m_header.m_tresolution));
    const int len_sft = n_p;
    
        // Calculate the optimal overlap
    int noverlap_optimal = get_optimal_overlap(len_sft)/ 2;
    printf("Optimal overlap: %i", noverlap_optimal);
   
       const int  Noverlap = pow(2, round(log2(noverlap_optimal)));
       if (m_nbin < 2 * Noverlap)
       {
           printf("nbin must be greater than %i", 2 * Noverlap);
       }

       const int LenChunk = (m_nbin - 2 * Noverlap) * m_nfft;
    // 4. memory allocation in GPU
   // total number of downloding bytes to each chunk:
       const long long QUantChunkBytes = calc_ChunkBytes(LenChunk);
    // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = calc_TotalChannelBytes();
    const long long QUantChunkComplexNumbers = calc_ChunkComplexNumbers();
    const long long QUantDownloadingBytesForChunk = calc_ChunkBytes(LenChunk);
    const long long QUantOverlapBytes= calc_ChunkBytes(Noverlap); 
    
   /* do_plan_and_memAlloc(&lenChunk
        ,&plan0, &plan1, &fdmt,&d_parrInput, &pcmparrRawSignalCur
        ,&pAuxBuff_fdmt,&d_arrfdmt_norm
        ,&pcarrTemp
        ,&pcarrCD_Out
        , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);*/
   
    void* parrInput =  nullptr;
    void* pcmparrRawSignalCur = nullptr;
    if (!(allocateInputMemory( &parrInput,  QUantDownloadingBytesForChunk,  &pcmparrRawSignalCur ,  QUantChunkComplexNumbers)))
    {
        return 1;
    }
    
    
    int ncoherent   = get_coherent_dms();
    
    CChunkB* pChunk= new  CChunkB();
    CChunkB** ppChunk = &pChunk;    
    createChunk(ppChunk
        , m_header.m_centfreq - m_header.m_obsBW / 2.0
        , m_header.m_centfreq + m_header.m_obsBW / 2.0
        , m_header.m_npol
        , m_header.m_nchan        
        , len_sft
        , 0
        , 0
        , m_d_max
        , m_d_min
        , ncoherent
        , m_sigma_bound
        , m_length_sum_wnd
        , m_nbin
        , m_nfft
        , Noverlap
       , m_header.m_tresolution );
    
   
    FILE** prb_File = (FILE**)malloc( sizeof(FILE*));
    if (!prb_File)
    {       
        return 1;
    }    
    openFileReadingStream(prb_File);    
   //std::vector<float>* pvecImg = new std::vector<float>;
  //  std::vector<std::vector<float>>* pvecImg = new std::vector<std::vector<float>>;
    for (int nB = 0; nB < IBlock; ++nB)        
    {          
        std::cout << "                               BLOCK=  " << nB <<std::endl;  
        createCurrentTelescopeHeader(prb_File);     
       
        const int NumChunks = (m_header.m_nblocksize - 2 *QUantOverlapBytes - 1) / (QUantChunkBytes - 2 *QUantOverlapBytes) + 1;
        // !6           
        for (int j = 0; j < NumChunks; ++j)
        { 
            if (j == (NumChunks - 1))
            {
                size_t shift = calc_ShiftingBytes(QUantChunkBytes);
                shift_file_pos(prb_File, -shift);
            }


           int ibytes = download_chunk(prb_File, (char*)parrInput, QUantChunkBytes);

            if (j != (NumChunks - 1))
            {
                size_t shift = calc_ShiftingBytes(2 * QUantOverlapBytes);
                shift_file_pos(prb_File, -shift);
            }           
            unpack_chunk(LenChunk, Noverlap,  (inp_type_*)parrInput,  pcmparrRawSignalCur);
           (*ppChunk)->set_blockid(nB);
           (*ppChunk)->set_chunkid(j);
           (*ppChunk)->process(pcmparrRawSignalCur, m_pvctSuccessHeaders, pvecImg);

           // TEMPORARY FOR DEBUGGING LOFAR ONLY! DELETE LATER!
           if (0== j)
           {
              break;
           }
               
        }
        *pmsamp = (*ppChunk)-> get_msamp();        

        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;
        //iTotal_time = duration.count();
        //std::cout << "Total time:                   " << iTotal_time << " microseconds" << std::endl;
        //std::cout << "FDMT time:                    " << iFdmt_time << " microseconds" << std::endl;
        //std::cout << "Read and data transform time: " << iReadTransform_time << " microseconds" << std::endl;
        //std::cout << "FFT time:                     " << iFFT_time << " microseconds" << std::endl;
        //std::cout << "Mean && disp time:            " << iMeanDisp_time << " microseconds" << std::endl;
        //std::cout << "Normalization time:           " << iNormalize_time << " microseconds" << std::endl;

        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;
        //std::cout << "/*****************************************************/ " << std::endl;	
        rewindFilePos(prb_File,   QUantTotalChannelBytes);            
    }
   closeFileReadingStream(prb_File);     
   freeInputMemory(parrInput, pcmparrRawSignalCur);    
    delete (*ppChunk);  
    ppChunk = nullptr;    
    return 0;
}
//-------------------------------------------------------------------
size_t  CSessionB::calc_ShiftingBytes(unsigned int QUantChunkBytes)
{
    return 0;
}
//------------------------------------------------------------------
void CSessionB::shift_file_ptr_to_last_pos(FILE** prb_File, const long long quant_bytes_per_polarization)
{   
}    
//--------------------------------------------------------------
void  CSessionB::shift_file_pos(FILE** prb_File,const int IShift)
{
 }
//---------------------------------------------------------------------------
void CSessionB::freeInputMemory(void* parrInput, void* pcmparrRawSignalCur)
{    
}
//-----------------------------------------------------------------
void CSessionB::createChunk(CChunkB** ppchunk
    , const float Fmin
    , const float Fmax
    , const int npol
    , const int nchan   
    , const unsigned int len_sft
    , const int Block_id
    , const int Chunk_id
    , const double d_max
    , const double d_min
    , const int ncoherent
    , const float sigma_bound
    , const int length_sum_wnd
    , const int nbin
    , const int nfft
    , const int noverlap
    , const float tsamp)
{   
}


bool CSessionB::allocateInputMemory(void** parrInput, const int QUantDownloadingBytesForChunk, void ** pcmparrRawSignalCur
,const int QUantChunkComplexNumbers) 
{
    return true;
}

//------------------------------------------------------------------
bool CSessionB::openFileReadingStream(FILE**& prb_File)
{
    return false;
}

//------------------------------------------------------------------
bool CSessionB::closeFileReadingStream(FILE**& prb_File)
{
    return false;
}
//-----------------------------------------------------------------------
// //-----------------------------------------------------------------
void CSessionB::rewindFilePos(FILE** prb_File, const int  QUantTotalChannelBytes)
{
}
//------------------------------------------------------------------------
bool CSessionB::readTelescopeHeader(FILE* r_File
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
    return false;
}
//-------------------------------------------------------------
bool CSessionB::createCurrentTelescopeHeader(FILE** prb_File)
{
    return false;
}
//---------------------------------------------------------
bool CSessionB::analyzeChunk(const COutChunkHeader outChunkHeader, CFragment* pFRg)
{ 
    //// calc quantity sessions
    //unsigned long long ilength = 0;

    //// 1. blocks quantity calculation

    //const int IBlock = calcQuantBlocks(&ilength);


    ////!1




    //// 3. cuFFT plans preparations
    //   // 3.1 reading first header




    //cufftHandle plan0 = NULL;
    //cufftHandle plan1 = NULL;
    //int lenChunk = 0;
    //char* d_parrInput = NULL;
    //cufftComplex* pcmparrRawSignalCur = NULL;
    //CFdmtU fdmt;
    //void* pAuxBuff_fdmt = NULL;
    //fdmt_type_* d_arrfdmt_norm = NULL;
    //cufftComplex* pcarrTemp = NULL; //2	
    //cufftComplex* pcarrCD_Out = NULL;//3
    //cufftComplex* pcarrBuff = NULL;//3
    //char* pInpOutBuffFdmt = NULL;
    //CChunk_cpu* pChunk = new CChunk_cpu();

    //do_plan_and_memAlloc(&lenChunk
    //    , &plan0, &plan1, &fdmt, &d_parrInput, &pcmparrRawSignalCur
    //    , &pAuxBuff_fdmt, &d_arrfdmt_norm
    //    , &pcarrTemp
    //    , &pcarrCD_Out
    //    , &pcarrBuff, &pInpOutBuffFdmt, &pChunk);

    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
    //    // Handle the error appropriately
    //}

    //// 4. memory allocation in GPU
    //// total number of downloding bytes to each chunk:
    //const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    //// total number of downloding bytes to each channel:
    //const long long QUantTotalChannelBytes = m_header.m_nblocksize * m_header.m_nbits / 8 / m_header.m_nchan;
    //const long long QUantChunkComplexNumbers = lenChunk * m_header.m_nchan * m_header.m_npol / 2;
    //FILE** prb_File = (FILE**)malloc(sizeof(FILE*));
    //if (!prb_File) {
    //    // Handle memory allocation failure
    //    return 1;
    //}

    //openFileReadingStream(prb_File);

    //
    //    
    //for (int nB = 0; nB < outChunkHeader.m_numBlock +1; ++nB)
    //{     

    //    cudaStatus = cudaGetLastError();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
    //        // Handle the error appropriately
    //    }

    //    createCurrentTelescopeHeader(prb_File);


    //    cudaStatus = cudaGetLastError();
    //    if (cudaStatus != cudaSuccess) {
    //        fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
    //        // Handle the error appropriately
    //    }
    //    // number of downloding bytes to each chunk:
    //    const long long QUantChunkBytes = lenChunk * m_header.m_nchan / 8 * m_header.m_npol * m_header.m_nbits;
    //    // total number of downloding bytes to each channel:
    //    const long long QUantTotalChannelBytes = m_header.m_nblocksize / m_header.m_nchan;
    //    // total number of downloading complex numbers of channel:
    //    const long long QUantChannelComplexNumbers = lenChunk * m_header.m_npol / 2;

    //    const int NumChunks = (m_header.m_nblocksize + QUantChunkBytes - 1) / QUantChunkBytes;

    //    // !6

    //    // 7. remains of not readed elements in block
    //    //long long iremainedBytes = m_header.m_nblocksize;
    //    float val_coherent_d;
    //    // !7
    //    float valSNR = -1;
    //    int argmaxRow = -1;
    //    int argmaxCol = -1;
    //    float coherentDedisp = -1.;

    //    for (int j = 0; j < NumChunks; ++j)
    //    {
    //    
    //        cudaStatus = cudaGetLastError();
    //        if (cudaStatus != cudaSuccess) {
    //            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
    //            // Handle the error appropriately
    //        }
    //        download_and_unpack_chunk(prb_File, lenChunk, j, (inp_type_*)d_parrInput, pcmparrRawSignalCur);
    //        long long position4 = ftell(*prb_File);

    //        if (!((nB == outChunkHeader.m_numBlock) && (j == outChunkHeader.m_numChunk)))
    //        {
    //            continue;
    //        }

    //        cudaStatus = cudaGetLastError();
    //        if (cudaStatus != cudaSuccess) {
    //            fprintf(stderr, "cudaGetLastError failed: %s\n", cudaGetErrorString(cudaStatus));
    //            // Handle the error appropriately
    //        }

    //        /*std::vector<inp_type_ > data0(quantDownloadingBytes/4, 0);
    //        cudaMemcpy(data0.data(), d_parrInput, quantDownloadingBytes / 4 * sizeof(inp_type_ ),
    //            cudaMemcpyDeviceToHost);
    //        cudaDeviceSynchronize();*/

    //        /*std::vector <std::complex<float>> data4(lenChunk* m_header.m_nchan * m_header.m_npol/2, 0);
    //        cudaMemcpy(data4.data(), pcmparrRawSignalCur, lenChunk * m_header.m_nchan * m_header.m_npol / 2 * sizeof(cufftComplex),
    //            cudaMemcpyDeviceToHost);
    //        cudaDeviceSynchronize();*/
    //                pChunk->detailedChunkProcessing(
    //                    outChunkHeader
    //                    , plan0
    //                    , plan1
    //                    , pcmparrRawSignalCur
    //                    , d_arrfdmt_norm
    //                    , pAuxBuff_fdmt
    //                    , pcarrTemp
    //                    , pcarrCD_Out
    //                    , pcarrBuff
    //                    , pInpOutBuffFdmt
    //                    , pFRg);

    //    }       
    //}
    //closeFileReadingStream(prb_File);
    //free(prb_File);
    //cudaFree(pcmparrRawSignalCur);
    //cudaFree(d_arrfdmt_norm);
    //cudaFree(pAuxBuff_fdmt);
    //cudaFree(pcarrTemp); //2	
    //cudaFree(pcarrCD_Out);//3
    //cudaFree(pcarrBuff);//3
    //cudaFree(pInpOutBuffFdmt);
    //cudaFree(d_parrInput);
    //delete pChunk;
    //cufftDestroy(plan0);
    //cufftDestroy(plan1);
    return 0;
 }

 //-----------------------------------------------------------------
 size_t  CSessionB::download_chunk(FILE** rb_file, char* d_parrInput, const long long QUantDownloadingBytes)
 {
     return 0;
 }
 //-----------------------------------------------------------------
 bool CSessionB::unpack_chunk(const long long lenChunk, const int j
     , inp_type_* d_parrInput, void* pcmparrRawSignalCur)
 {
     return false;
 }
//------------------------------------
bool CSessionB::navigateToBlock(FILE* rb_File,const int IBlockNum)
{  
    return true;
}


//
////-----------------------------------------------------------------
//bool CSessionB::do_plan_and_memAlloc( int* pLenChunk
//    , cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
//    , void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
//    , cufftComplex** ppcarrTemp
//    , cufftComplex** ppcarrCD_Out
//    , cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt, CChunk_cpu** ppChunk)
//
//{
//    // 2.allocation memory for parametrs
//  //  in CPU
//    int nbits = 0;
//    float chanBW = 0;
//    int npol = 0;
//    bool bdirectIO = 0;
//    float centfreq = 0;
//    int nchan = 0;
//    float obsBW = 0;
//    long long nblocksize = 0;
//    EN_telescope TELESCOP = GBT;
//    int ireturn = 0;
//    float tresolution = 0.;
//    // !2
//    FILE* rb_File = fopen(m_strInpPath, "rb");
//    if (!readTelescopeHeader(
//        rb_File
//        , &nbits
//        , &chanBW
//        , &npol
//        , &bdirectIO
//        , &centfreq
//        , &nchan
//        , &obsBW
//        , &nblocksize
//        , &TELESCOP
//        , &tresolution
//    )
//        )
//    {
//        return false;
//    }
//    fclose(rb_File);
//    if (nbits / 8 != sizeof(inp_type_))
//    {
//        std::cout << "check up Constants.h, inp_type_  " << std::endl;
//        return -1;
//    }
//
//    m_header = CTelescopeHeader(
//        nbits
//        , chanBW
//        , npol
//        , bdirectIO
//        , centfreq
//        , nchan
//        , obsBW
//        , nblocksize
//        , TELESCOP
//        , tresolution
//    );
//   
//
//    plan_and_memAlloc( pLenChunk
//        , pplan0, pplan1, pfdmt, d_pparrInput, ppcmparrRawSignalCur
//        , ppAuxBuff_fdmt, d_parrfdmt_norm, ppcarrTemp, ppcarrCD_Out
//        , ppcarrBuff, ppInpOutBuffFdmt, ppChunk);
//}
//
////------------------------------------------------------------------------------
//
//void CSessionB::plan_and_memAlloc(
//    int* pLenChunk
//    , cufftHandle* pplan0, cufftHandle* pplan1, CFdmtU* pfdmt, char** d_pparrInput, cufftComplex** ppcmparrRawSignalCur
//    , void** ppAuxBuff_fdmt, fdmt_type_** d_parrfdmt_norm
//    , cufftComplex** ppcarrTemp
//    , cufftComplex** ppcarrCD_Out
//    , cufftComplex** ppcarrBuff, char** ppInpOutBuffFdmt, CChunk_cpu** ppChunk)
//{
//    cudaError_t cudaStatus;
//    const float VAlFmin =  m_header.m_centfreq - ((float) m_header.m_nchan) *  m_header.m_chanBW / 2.0;
//    const float VAlFmax =  m_header.m_centfreq + ((float) m_header.m_nchan) *  m_header.m_chanBW / 2.0;
//    // 3.2 calculate standard len_sft and LenChunk    
//    const int len_sft = calc_len_sft(fabs( m_header.m_chanBW), m_pulse_length);
//    *pLenChunk = _calcLenChunk_( m_header, len_sft, m_pulse_length, m_d_max);
//
//
//    // 3.3 cuFFT plans preparations
//
//    cufftCreate(pplan0);
//    checkCudaErrors(cufftPlan1d(pplan0, *pLenChunk, CUFFT_C2C,  m_header.m_nchan *  m_header.m_npol / 2));
//
//
//
//    cufftCreate(pplan1);
//    checkCudaErrors(cufftPlan1d(pplan1, len_sft, CUFFT_C2C, (*pLenChunk) *  m_header.m_nchan *  m_header.m_npol / 2 / len_sft));
//
//
//
//    // !3
//
//    // 4. memory allocation in GPU
//    // total number of downloding bytes to each file:
//    const long long QUantDownloadingBytesForChunk = (*pLenChunk) *  m_header.m_nchan / 8 *  m_header.m_nbits *  m_header.m_npol;
//
//    const long long QUantBlockComplexNumbers = (*pLenChunk) *  m_header.m_nchan *  m_header.m_npol / 2;
//
//
//
//    checkCudaErrors(cudaMallocManaged((void**)d_pparrInput, QUantDownloadingBytesForChunk * sizeof(char)));
//
//
//    checkCudaErrors(cudaMalloc((void**)ppcmparrRawSignalCur, QUantBlockComplexNumbers * sizeof(cufftComplex)));
//    // 2!
//
//
//
//    // 4.memory allocation for auxillary buffer for fdmt   
//       // there is  quantity of real channels
//    const int NChan_fdmt_act = len_sft *  m_header.m_nchan;
//    (*pfdmt) = CFdmtU(
//        VAlFmin
//        , VAlFmax
//        , NChan_fdmt_act
//        , (*pLenChunk) / len_sft
//        , m_pulse_length
//        , m_d_max
//        , len_sft);
//
//
//
//    size_t szBuff_fdmt = pfdmt->calcSizeAuxBuff_fdmt_();
//
//    checkCudaErrors(cudaMalloc(ppAuxBuff_fdmt, szBuff_fdmt));
//    // 4!
//
//
//    // 3. memory allocation for fdmt_ones on GPU  ????
//    size_t szBuff_fdmt_output = pfdmt->calc_size_output();
//
//    checkCudaErrors(cudaMalloc((void**)d_parrfdmt_norm, szBuff_fdmt_output));
//    //// 6. calculation fdmt ones
//    pfdmt->process_image(nullptr      // on-device input image
//        , *ppAuxBuff_fdmt
//        , *d_parrfdmt_norm	// OUTPUT image,
//        , true);
//
//    // 3!
//
//
//
//
//    // 5. memory allocation for the 3 auxillary cufftComplex  arrays on GPU	
//    //cufftComplex* pffted_rowsignal = NULL; //1	
//
//
//
//    checkCudaErrors(cudaMalloc((void**)ppcarrTemp, QUantBlockComplexNumbers * sizeof(cufftComplex)));
//
//    checkCudaErrors(cudaMalloc((void**)ppcarrCD_Out, QUantBlockComplexNumbers * sizeof(cufftComplex)));
//
//    checkCudaErrors(cudaMalloc((void**)ppcarrBuff, QUantBlockComplexNumbers * sizeof(cufftComplex)));
//    // !5
//
//    // 5. memory allocation for the 2 auxillary arrays on GPU for input and output of FDMT	
//    size_t szInpOut_fdmt = pfdmt->calc_size_output() + pfdmt->calc_size_input();
//
//    checkCudaErrors(cudaMalloc((void**)ppInpOutBuffFdmt, szInpOut_fdmt));
//
//    // 5!
//
//    // !4	
//    **ppChunk = CChunk_cpu(
//        VAlFmin
//        , VAlFmax
//        ,  m_header.m_npol
//        ,  m_header.m_nchan
//        , (*pLenChunk)
//        , len_sft
//        , 0
//        , 0
//        ,  m_header.m_nbits
//        , m_d_max
//        , m_sigma_bound
//        , m_length_sum_wnd
//        , *pfdmt
//        , m_pulse_length
//    );
//
//}
////------------------------------------------------------------
//
//long long CSessionB::_calcLenChunk_(CTelescopeHeader header, const int nsft
//    , const float pulse_length, const float d_max)
//{
//    const int nchan_actual = nsft * header.m_nchan;
//
//    long long len = 0;
//    for (len = 1 << 9; len < 1 << 30; len <<= 1)
//    {
//        CFdmtU fdmt(
//            header.m_centfreq - header.m_chanBW * header.m_nchan / 2.
//            , header.m_centfreq + header.m_chanBW * header.m_nchan / 2.
//            , nchan_actual
//            , len
//            , pulse_length
//            , d_max
//            , nsft
//        );
//        long long size0 = fdmt.calcSizeAuxBuff_fdmt_();
//        long long size_fdmt_inp = fdmt.calc_size_input();
//        long long size_fdmt_out = fdmt.calc_size_output();
//        long long size_fdmt_norm = size_fdmt_out;
//        long long irest = header.m_nchan * header.m_npol * header.m_nbits / 8 // input buff
//            + header.m_nchan * header.m_npol / 2 * sizeof(cufftComplex)
//            + 3 * header.m_nchan * header.m_npol * sizeof(cufftComplex) / 2
//            + 2 * header.m_nchan * sizeof(float);
//        irest *= len;
//
//        long long rez = size0 + size_fdmt_inp + size_fdmt_out + size_fdmt_norm + irest;
//        if (rez > 0.98 * TOtal_GPU_Bytes)
//        {
//            return len / 2;
//        }
//
//    }
//    return -1;
//}

int CSessionB::get_optimal_overlap(const int nsft)
{
    float  bw_chan = m_header.m_obsBW / (m_header.m_nchan * nsft);       
    float   fmin_bottom = m_header.m_centfreq - m_header.m_obsBW / 2.;        
    float   fmin_top = fmin_bottom + bw_chan;
    float   delay = 4.148808e3 * m_d_max * (1.0 / (fmin_bottom * fmin_bottom) - 1.0 / (fmin_top * fmin_top));
    float   delay_samples = round(delay / m_header.m_tresolution);
    return int(delay_samples);
}
//--------------------------------------------------------------
int  CSessionB::get_coherent_dms()
{
    // Compute the coherent DMs for the FDMT algorithm.
    float f_min = m_header.m_centfreq - m_header.m_obsBW / 2.0; 
    float f_max = m_header.m_centfreq + m_header.m_obsBW / 2.0;
    float    t_d = 4.148808e3 * m_d_max * (1.0 / (f_min * f_min) - 1.0 / (f_max * f_max));
    int irez = ceil(t_d * m_header.m_tresolution / (m_pulse_length * m_pulse_length));
    return irez;
}