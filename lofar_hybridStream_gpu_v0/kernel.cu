// ./kernel -n 8192 -N 32768 -P 001 -b 4 -d 0.5 1.0 80 -o OutImages -q L2012176_SAP000_B000_S0_P001_bf.h5
// .\lofar_hybridStream_gpu_v0.exe -P 25.0E-6 -b 12.0 -d 10 -k 80.0 -N OutPutFold -n D:/BASSA/hdf5_data/L2012176_SAP000_B000_S0_P001_bf.h5


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <limits.h>
#include <stdint.h>

#include "kernel.cuh"
#include "npy.hpp"
#include "Constants.h"
#include "LofarSession.cuh"
#include "OutChunkHeader.h"
#include "Fragment.h"


#ifdef _WIN32 // Windows
#include "DrawImg.h"
#else // Linux

#endif  




using namespace std;




char PathLofarFile[] = "D://BASSA//hdf5_data//L2012176_SAP000_B000_S0_P001_bf.h5";
char PathOutFold[] = "OutPutFold";

//
void showInputString(char* pPathLofarFile, char* pPathOutFold, float length_of_pulse
    , float VAlD_max, float sigma_Bound, int lenWindow)
{
    std::cout << "pPathLofarFile:    " << pPathLofarFile << std::endl;
    std::cout << "pPathOutFold:      " << pPathOutFold << std::endl;
    std::cout << "length_of_pulse =  " << length_of_pulse << std::endl;
    std::cout << "VAlD_max =         " << VAlD_max << std::endl;
    std::cout << "sigma_Bound =      " << sigma_Bound << std::endl;
    std::cout << "lenWindow =        " << lenWindow << std::endl;
}
int main(int argc, char** argv)
{   
  
    //defining by default input parameters of cmd line                                                              
    char* pPathLofarFile = PathLofarFile;
    char* pPathOutFold = PathOutFold;
    double valD_max = 80.0;
    double length_of_pulse = 5.12E-6 * 8.0;//;
    float sigma_Bound = 12.;
    int lenWindow = 10;
    // !

    if (argc > 1)
    {
        if (argc < 11)
        {
            std::cerr << "Usage: " << argv[0] << " -n <InpFile> -N <OutFold> -P <length_of_pulse> -b <tresh> -d <lenWin>" << std::endl;
            return 1;
        }
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "-n")
            {
                pPathLofarFile = argv[++i];
                continue;
            }
            if (std::string(argv[i]) == "-N")
            {
                pPathOutFold = argv[++i];
                continue;
            }
            if (std::string(argv[i]) == "-P")
            {
                length_of_pulse = std::atof(argv[++i]);
                continue;
            }
            if (std::string(argv[i]) == "-b")
            {
                sigma_Bound = std::atof(argv[++i]);
                continue;
            }

            if (std::string(argv[i]) == "-k")
            {
                valD_max = std::atof(argv[++i]);
                continue;
            }
            if (std::string(argv[i]) == "-d")
            {
                lenWindow = std::atoi(argv[++i]);
                continue;
            }           
        }
    }
    showInputString(pPathLofarFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow);

    CLofarSession* pSession = new CLofarSession(pPathLofarFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow);
    unsigned long long ilength = 0;
    //int iBlocks = pSession->calcQuantRemainBlocks(&ilength);
                           
    if (-1 == pSession->launch())
    {
        delete pSession;
        return -1;
    }

    if (pSession->m_pvctSuccessHeaders->size() > 0)
    {
        std::cout << "               Successful Chunk Numbers = " << pSession->m_pvctSuccessHeaders->size() << std::endl;
        //--------------------------------------

        char charrTemp[200] = { 0 };
        for (int i = 0; i < pSession->m_pvctSuccessHeaders->size(); ++i)
        {
            memset(charrTemp, 0, 200 * sizeof(char));
            (*(pSession->m_pvctSuccessHeaders))[i].createOutStr(charrTemp);
            std::cout << i+1<<". "<< charrTemp << std::endl;
        }
        //--------------------------------------
        
    }
    else
    {
        std::cout << "               Successful Chunk Were Not Detected= " << std::endl;
        
        return 0;

    }

    char outputlogfile[300] = { 0 };
    strcpy(outputlogfile, pPathOutFold);
    strcat(outputlogfile, "//output.log");
    COutChunkHeader::writeReport(outputlogfile, pSession->m_pvctSuccessHeaders
        , length_of_pulse);
    delete pSession;

    char chInp[200] = { 0 };
    std::cout << "if you  want to quit, print q" << endl;
    std::cout << "if you want to proceed, print y " << endl;

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    char ch0 = std::cin.get();
    if (ch0 == 'q')
    {
        return 0;
    }
    ////----------------------------------------------------------------------------------------------------
        // SECOND PART - PAINTINGS
    //----------------------------------------------------------------------------------------------------
    std::cout << "print number of chunk: " << endl;
    int numOrder = -1;
    std::cin >> numOrder;
    --numOrder;
   
    int numBlock = -1;
    int numChunk = -1;
    long long lenChunk = -1;
    int n_fdmtRows = -1, n_fdmtCols = -1;
    int sucRow = -1, sucCol = -1, width = -1;
    float cohDisp = -1., snr = -1.;
    
    COutChunkHeader::read_outputlogfile_line(outputlogfile
        , numOrder
        , &numBlock
        , &numChunk
        , &n_fdmtRows
        , &n_fdmtCols
        , &sucRow
        , &sucCol
        , &width
        , &cohDisp
        , &snr);
    //---------
    COutChunkHeader outChunkHeader(
        n_fdmtRows
        , n_fdmtCols
        , sucRow
        , sucCol
        , width
        , snr
        , cohDisp
        , numBlock-1
        , numChunk-1
    );
   
    pSession = new CLofarSession(pPathLofarFile, pPathOutFold, length_of_pulse, valD_max, sigma_Bound, lenWindow);

    CFragment *pFRg = new CFragment();
    pSession->analyzeChunk(outChunkHeader,  pFRg);

     
    int dim = pFRg->m_dim;   
    delete pSession;

    std::array<long unsigned, 2> leshape101{ dim, dim };

    npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), pFRg->m_vctData);

    std::vector<float>v = pFRg->m_vctData; 
    delete pFRg;

#ifdef _WIN32 // Windows
    char filename_gpu[] = "image_gpu.png";
    createImg_(argc, argv, v, dim, dim, filename_gpu);
#else
#endif 

   
    return 0;
}

