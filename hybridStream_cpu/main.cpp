#include "main.h"
#include <iostream>
//#include <complex>
#include <fftw3.h>
#include <GL/glut.h>
#include <fstream>
#include "HybridC_v0.h"
#include "utilites.h"
#include "StreamParams.h"

#include <array>
#include <iostream>
#include <string>

#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 
#include "read_and_write_log.h"
#include <chrono>
#include "fileInput.h"
#include "DrawImg.h"
#include "Constants.h"


#define _CRT_SECURE_NO_WARNINGS
using namespace std;

class StreamParams;


const char chStrDefaultInputPass[] = "..//HYBRID_TESTS//data_.bin";
//const char chStrDefaultInputPass[] = "..//HYBRID_TESTS//data3.bin";
int numAttemptions = 0;

int dialIntroducing()
{
    // 1. define path to data file with complex time serie
    std::cout << "By default input file is  " << chStrDefaultInputPass << endl;//  \"D://MyVSprojPy//hybrid//data.bin\"" << endl;
    std::cout << "if you want default, print y, otherwise n" << endl;
    char userInput[200] = { 0 };
    char ch = std::cin.get();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
   
    
    if (ch == 'y')
    {       
        strcpy(mchInpFilePass, chStrDefaultInputPass);
    }
    else
    {
        std::cout << "Enter the pass:" << endl;//  with double quotation marks \"..\"" << endl;
        std::cin.getline(userInput, 200);
        strcpy(mchInpFilePass, userInput);
    }
    // 1!


    // 2. reading header of input file
   

    if (readHeader(mchInpFilePass, mlenarr, m_n_p
        , mvalD_max, mvalf_min, mvalf_max, mvalSigmaBound) == 1)
    {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    // 2 !

    // 3. printing header's info
    std::cout << "Header's information:" << endl;
    std::cout << "Length of time serie = " << mlenarr << endl;
    // 3!

    // 4. default parametres
    
    mnumBegin = 1;
    mnumEnd = mlenarr ;
    mlenChunk = pow(2, 20);
    std::cout << "By default parametres:" << endl;
    std::cout << "Length of chunk( 2 **20 ) = " << mlenChunk << endl;
    std::cout << "Number of first elem =  " << mnumBegin << endl;
    std::cout << "Number of last elem =  " << mnumEnd << endl;
    std::cout << "If you want go on by default print y, otherwise print n " << endl;
    
    char ch1 = std::cin.get();
    if (ch1 != 'y')
    {
        for (int i = 0; i < 4; ++i)
        {
            std::cout << "Print begin number of time serie: ";
            std::cin >> mnumBegin;

            std::cout << "Print end number of time serie: ";
            std::cin >> mnumEnd;

            std::cout << "Print chunk's length: ";
            std::cin >> mlenChunk;

            if ((mnumBegin < 1) || (mnumEnd > mlenarr) || (mlenChunk > (mnumEnd - mnumBegin)))
            {
                std::cout << "Check up parametres" << endl;
                ++numAttemptions;
                if (numAttemptions == 4)
                {
                    return 2;
                }
            }
            else
            {
                break;
            }
        }
    }
    // 4!
    mnumBegin -= 1;
    mnumEnd -= 1;
    return 0;
}

int main(int argc, char** argv)
{  
    dialIntroducing();
    // 5. Create member of class CStreamParams
    CStreamParams* pStreamPars = new CStreamParams(mchInpFilePass, mnumBegin, mnumEnd, mlenChunk);
    // 5!
    
    // 6. memory allocation for output information 
    int* piarrNumSucessfulChunks = (int*)malloc(sizeof(int) *(1 + (mnumEnd - mnumBegin)/ mlenChunk));
    float* parrCoherent_d = (float*)malloc(sizeof(float) * (1 + (mnumEnd - mnumBegin) / mlenChunk));
    int quantOfSuccessfulChunks = 0;    
    // 6 !
   
    // 7. call function processing the stream    
    int irez = fncHybridScan(nullptr, piarrNumSucessfulChunks, parrCoherent_d, quantOfSuccessfulChunks, pStreamPars);

    fncWriteLog_("info.log", mchInpFilePass, "hybrid dedispersion, C++ implementation"
        , mlenChunk, quantOfSuccessfulChunks, piarrNumSucessfulChunks, parrCoherent_d,0);

    // 7!
    
    // 8. report
    std::cout << "------------ Calculations completed successfully -------------" << endl;
    std::cout << "Pass to Data File : " << mchInpFilePass << endl;
    std::cout << "Successful Chunks Number : " << quantOfSuccessfulChunks<< endl;
    std::cout << "Chunk Num., Coh. Disp. : " << endl;
    for (int i = 0; i < quantOfSuccessfulChunks; ++i)
    {
        std::cout <<i +1<<") : "<< piarrNumSucessfulChunks[i] << " ; " << parrCoherent_d[i] << endl;
    }

    free(piarrNumSucessfulChunks);
    free(parrCoherent_d);    
    delete pStreamPars;

    std::cout << "Running Time = " << 0. << "ms"<<endl;
    std::cout << "---------------------------------------------------------" << endl;   
    
    char chInp[200] = { 0 };
    std::cout << "if you  want to quit, print q" << endl;
    std::cout << "if you want to proceed, print y " << endl;

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    char ch0 = std::cin.get();
    if (ch0 == 'q')
    {
        return 0;
    }
//----------------------------------------------------------------------------------------------------
    // SECOND PART
//----------------------------------------------------------------------------------------------------
    std::cout << "print number of chunk: " << endl;
    int numOrder = -1;
    std::cin >> numOrder;
    --numOrder;
    char strPassLog[] = "info.log";
    int lengthOfChunk = 0, quantChunks = 0;
    int arrChunks[1000] = { 0 };
    float arrCohD[1000] = { 0. };
    char strPassDataFile[200] = { 0 };
    
    fncReadLog_(strPassLog, strPassDataFile, &lengthOfChunk, &quantChunks, arrChunks, arrCohD);
    unsigned int lenarr1 = 0, n_p1 = 0;
    float valD_max1 = 0., valf_min1 = 0., valf_max1 = 0., valSigmaBound1 = 0.;

    if (readHeader(strPassDataFile, lenarr1, n_p1
        , valD_max1, valf_min1, valf_max1, valSigmaBound1) == 1)
    {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    const int NUmChunk = arrChunks[numOrder];
    const float VAlCohD = arrCohD[numOrder];
    CStreamParams StreamPars1(strPassDataFile, NUmChunk * lengthOfChunk, (NUmChunk + 1) * lengthOfChunk,
        lengthOfChunk);
    
    // create output numpy files with images
   
    
    float*  poutputImage = (float*)malloc((StreamPars1.m_n_p) * (StreamPars1.m_lenChunk / StreamPars1.m_n_p)
            * sizeof(float));
    float* poutputPartImage = (float*)malloc( sizeof(float));
    float** ppoutputPartImage = &poutputPartImage;   


    int  iargmaxCol = -1, iargmaxRow = -1;
    float valSNR = -1;
    int quantRowsPartImage = -1;
    createOutImageForFixedNumberChunk(poutputImage,&iargmaxRow, &iargmaxCol, &valSNR, ppoutputPartImage,&quantRowsPartImage, &StreamPars1, NUmChunk, VAlCohD); 

    std::cout << "OUTPUT DATA: " << endl;
    std::cout << "CHUNK NUMBER = " << NUmChunk <<endl;
    std::cout << "SNR = " << valSNR << endl;
    std::cout << "ROW = " << iargmaxRow << endl;
    std::cout << "COLUMN  = " << iargmaxCol << endl;
        
    std::vector<float> v1(poutputPartImage, poutputPartImage + quantRowsPartImage * quantRowsPartImage);

    std::array<long unsigned, 2> leshape101 { quantRowsPartImage, quantRowsPartImage };

    npy::SaveArrayAsNumpy("out_image.npy", false, leshape101.size(), leshape101.data(), v1);    
         
    ppoutputPartImage = nullptr;
    
    free(poutputImage);
    free(poutputPartImage);   
   

    char filename_cpu[] = "image_cpu.png";
    createImg_(argc, argv, v1, quantRowsPartImage, quantRowsPartImage, filename_cpu);

    return 0;    
}
