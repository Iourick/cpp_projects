#include "SessionB.h"
#include <string>
#include "stdio.h"
#include <iostream>
#include "OutChunkHeader.h"
#include <vector>


//#include "CFdmtC.h"
#include <stdlib.h>
#include <fftw3.h>
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
// OUTPUT:
// * pvecImg - output fdmt image
// *pmsamp - quantuty time-domain samples in chunk
int CSessionB::launch(std::vector<std::vector<float>>* pvecImg, int *pmsamp)
{     
    // 1. blocks quantity calculation  
    // In GUPPI case there can be multiple blocks, in LOFAR case the only 1 
    unsigned long long ilength = 0;
    const int IBlock = calcQuantBlocks(&ilength);     
    //!1 
    
    // 2. reading raw data header, overloaded for GUPPI and LOFAR cases
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
    // !2 


    if (m_header.m_nbits / 8 != sizeof(inp_type_))
    {
        std::cout << "check up Constants.h, inp_type_  " << std::endl;
        return -1;
    }  

    // 3. Calculate the optimal channels per subband
    const int n_p = int(ceil(m_pulse_length / m_header.m_tresolution));
    const int len_sft = n_p;
    // !3
    
    // 4. Calculate the optimal overlap
    int noverlap_optimal = get_optimal_overlap(len_sft)/ 2;
    printf("Optimal overlap: %i", noverlap_optimal);
   
    const int  Noverlap = pow(2, round(log2(noverlap_optimal)));
    if (m_nbin < 2 * Noverlap)
    {
        printf("nbin must be greater than %i", 2 * Noverlap);
    }
    // !4

    // 5. calculation lengt of processing time series
    const int LenChunk = (m_nbin - 2 * Noverlap) * m_nfft;
     // !5
       
    // 6. calculation constants for memory managment
        // total number of downloding bytes to each chunk:
    const long long QUantChunkBytes = calc_ChunkBytes(LenChunk);
      // total number of downloding bytes to each channel:
    const long long QUantTotalChannelBytes = calc_TotalChannelBytes();
    const long long QUantChunkComplexNumbers = calc_ChunkComplexNumbers();
    const long long QUantDownloadingBytesForChunk = calc_ChunkBytes(LenChunk);
    const long long QUantOverlapBytes= calc_ChunkBytes(Noverlap);     
  // !6
   
    // 7. Memory allocation for input buffer
    // overloaded 4 times- {LOFAR, GUPPI}x{CPU,GPU}
    void* parrInput =  nullptr;
    void* pcmparrRawSignalCur = nullptr;
    if (!(allocateInputMemory( &parrInput,  QUantDownloadingBytesForChunk,  &pcmparrRawSignalCur ,  QUantChunkComplexNumbers)))
    {
        return 1;
    }
    // !7
    
    // 8.  Compute the coherent DMs for the FDMT algorithm.
    int ncoherent   = get_coherent_dms();
    //!8
    
    // 9. Chunk creation. Overloaded for CPU and GPU cases
    CChunkB* pChunk= new  CChunkB();
    CChunkB** ppChunk = &pChunk;    
    createChunk(ppChunk
        , m_header.m_centfreq - fabs(m_header.m_obsBW) / 2.0
        , m_header.m_centfreq + fabs(m_header.m_obsBW) / 2.0
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
    // !9   
   
    // 10. Opening raw data reading session, overloaded
    FILE** prb_File = (FILE**)malloc( sizeof(FILE*));
    if (!prb_File)
    {       
        return 1;
    }    
    openFileReadingStream(prb_File);    
    // !10
   
    /*--------  11. Main loop, for each block in session -----------------------------------------------------------------------------------------------------------------------*/ 
    for (int nB = 0; nB < IBlock; ++nB)        
    { 
        //11.1 creation current telescope header. GUPPi raw file contains header for each block, contrary to LOFAR
       createCurrentTelescopeHeader(prb_File);  
       // !11.1
       
       // 11.2 calculation chunk's number in block. 
        const int NumChunks = (m_header.m_nblocksize - 2 *QUantOverlapBytes - 1) / (QUantChunkBytes - 2 *QUantOverlapBytes) + 1;
        std::cout << "    BLOCK=  " << nB <<  "  NUM CHUNKS = "<< NumChunks<<std::endl;
        // !11.2     

        // !11.3 Loop for each chunk in block          
        for (int j = 0; j < NumChunks; ++j)
        { 
            std::cout << "                chunk =   " << j <<  std::endl;

            // 11.4 arrangements with last chunk in block. can not me multiple.
            if (j == (NumChunks - 1))
            {
                size_t shift = calc_ShiftingBytes(QUantChunkBytes);
                shift_file_pos(prb_File, -shift);
            }
            // !11.4

            // 11.5 downloading chunk in input buffer
           int ibytes = download_chunk(prb_File, (char*)parrInput, QUantChunkBytes);
           // !11.5

           // 11.6 preparation file pointer's true positon for next downloading
            if (j != (NumChunks - 1))
            {
                size_t shift = calc_ShiftingBytes(2 * QUantOverlapBytes);
                shift_file_pos(prb_File, -shift);
            }    
            // !11.6

            // 11.7 unpack_chunk(..) is overloaded for 4 cases: {LOFAR, GUPPI} x  {CPU,GPU}
            // type of raw data file, place of memory allocation
            // pcmparrRawSignalCur - unpacked complex raw signal
            unpack_chunk(LenChunk, Noverlap,  (inp_type_*)parrInput,  pcmparrRawSignalCur);
            // !11.7
           (*ppChunk)->set_blockid(nB);
           (*ppChunk)->set_chunkid(j);

           // 11.8 Chunk's processing. Overloaded for 2 cases - CPU and GPU
           // m_pvctSuccessHeaders - is not being used
           // pvecImg - ouput fdmt vector
           // pcmparrRawSignalCur - input complex raw signal
           (*ppChunk)->process(pcmparrRawSignalCur, m_pvctSuccessHeaders, pvecImg);
           // !11.8          
        }
       
        *pmsamp = (*ppChunk)-> get_msamp();        

        rewindFilePos(prb_File,   QUantTotalChannelBytes);            
    }
   /*--------------  !11   -------------------------------------------------------------------------------------------------------------*/


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
//------------------------------------------------------------------------------------------
int CSessionB::get_optimal_overlap(const int nsft)
{
    float  bw_chan = fabs(m_header.m_obsBW) / (m_header.m_nchan * nsft);       
    float   fmin_bottom = m_header.m_centfreq - fabs(m_header.m_obsBW) / 2.;
    float   fmin_top = fmin_bottom + bw_chan;
    float   delay = 4.148808e3 * m_d_max * (1.0 / (fmin_bottom * fmin_bottom) - 1.0 / (fmin_top * fmin_top));
    float   delay_samples = round(delay / m_header.m_tresolution);
    return int(delay_samples);
}
//--------------------------------------------------------------
// Compute the coherent DMs for the FDMT algorithm.
int  CSessionB::get_coherent_dms()
{    
    float f_min = m_header.m_centfreq - fabs(m_header.m_obsBW) / 2.0;
    float f_max = m_header.m_centfreq + fabs(m_header.m_obsBW) / 2.0;
    float    t_d = 4.148808e3 * m_d_max * (1.0 / (f_min * f_min) - 1.0 / (f_max * f_max));
    int irez = ceil(t_d * m_header.m_tresolution / (m_pulse_length * m_pulse_length));
    return irez;
}