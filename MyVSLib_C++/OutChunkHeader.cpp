#include "OutChunkHeader.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>

COutChunkHeader::~COutChunkHeader() = default;

COutChunkHeader::COutChunkHeader()
{
	m_nrows = 0;
	m_ncols = 0;
	m_nSucessRow = 0;
	m_nSucessCol = 0;
	m_SNR = 0;
	m_coherentDedisp = 0.;
	m_numBlock = 0;
	m_numChunk = 0;
	m_wnd_width = 0.;
}
//-----------------------------------------------------------

COutChunkHeader::COutChunkHeader(const  COutChunkHeader& R)
{
	m_nrows = R.m_nrows;
	m_ncols = R.m_ncols;
	m_nSucessRow = R.m_nSucessRow;
	m_nSucessCol = R.m_nSucessCol;
	m_SNR = R.m_SNR;
	m_coherentDedisp = R.m_coherentDedisp;
	m_numBlock = R.m_numBlock;
	m_numChunk = R.m_numChunk;
	m_wnd_width = R.m_wnd_width;
}
//-------------------------------------------------------------------

COutChunkHeader& COutChunkHeader::operator=(const COutChunkHeader& R)
{
	if (this == &R)
	{
		return *this;
	}

	m_nrows = R.m_nrows;
	m_ncols = R.m_ncols;
	m_nSucessRow = R.m_nSucessRow;
	m_nSucessCol = R.m_nSucessCol;
	m_SNR = R.m_SNR;
	m_coherentDedisp = R.m_coherentDedisp;
	m_numBlock = R.m_numBlock;
	m_numChunk = R.m_numChunk;
	m_wnd_width = R.m_wnd_width;
	return *this;
}
//------------------------------------------------------------------
COutChunkHeader::COutChunkHeader(
	const int nrows
	, const int ncols
	, const int nSucessRow
	, const int nSucessCol
	, const int wnd_width
	, const float SNR
	, const float 	coherentDedisp
	, const int numBlock
	, const int numChunk
)

{
	m_nrows = nrows;

	m_ncols = ncols;

	m_nSucessRow = nSucessRow;

	m_nSucessCol = nSucessCol;

	m_SNR = SNR;

	m_coherentDedisp = coherentDedisp;

	m_numBlock = numBlock;

	m_numChunk = numChunk;

	m_wnd_width = wnd_width;

}
//-----------------------------------------------
void COutChunkHeader::createOutStr(char* pstr)
{	
	sprintf(pstr, "Block=  %d,Chunk=  %d,Rows=  %d,Cols=  %d, SucRow=  %d,SucCol=  %d, SNR=  %.4f, CohDisp=  %.4f, windWidth=  %d", m_numBlock +1
		, m_numChunk +1, m_nrows, m_ncols, m_nSucessRow , m_nSucessCol, m_SNR, m_coherentDedisp, m_wnd_width);
}
//-------------------------------------------------
bool COutChunkHeader::read_outputlogfile_line(const char* pstrPassLog
    , const int NUmLine
    , int* pnumBlock
    , int* pnumChunk
    , int* pn_fdmtRows
    , int* n_fdmtCols
    , int* psucRow
    , int* psucCol
    , int* pwidth
    , float* pcohDisp
    , float* snr
)
{
    //1. download enough data
    char line[300] = { 0 };

    FILE* fp = fopen(pstrPassLog, "r");
    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", pstrPassLog);
        return EXIT_FAILURE;
    }

    /* Get the first line of the file. */
    for (int i = 0; i < NUmLine + 1; ++i)
    {
        fgets(line, 300, fp);
    }
    fclose(fp);
    //2. check up mode. if mode != RAW return false  
    char* p = strstr(line, "Block=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pnumBlock = atoi(p + 8);

    p = strstr(line, "Chunk=");
    if (NULL == p)
    {
        return false;
    }

    *pnumChunk = atoi(p + 8);

    p = strstr(line, "Rows=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pn_fdmtRows = atoi(p + 7);

    p = strstr(line, "Cols=");
    if (NULL == p)
    {
        return false;
    }
    *n_fdmtCols = atoi(p + 7);

    p = strstr(line, "SucRow=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *psucRow = atoi(p + 9);

    p = strstr(line, "SucCol=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *psucCol = atoi(p + 9);

    p = strstr(line, "SNR=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *snr = atof(p + 6);

    p = strstr(line, "CohDisp=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pcohDisp = atof(p + 10);

    p = strstr(line, "windWidth=");
    if (NULL == p)
    {
        delete p;
        return false;
    }
    *pwidth = atoi(p + 12);
    return true;
}
//----------------------------------------
   
void COutChunkHeader::writeReport(const char* chstrOutput, std::vector<COutChunkHeader>* pvctSuccessHeaders
    ,const float pulse_length)
{
    FILE* wb_file = NULL;
    if ((wb_file = fopen(chstrOutput, "wb")) == NULL)
    {
        printf("Can not open output file for writing");
        return;
    }
    for (int i = 0; i < pvctSuccessHeaders->size(); ++i)
    {
        char arrch[2000] = { 0 };
        char charrTemp[200] = { 0 };

        (*pvctSuccessHeaders)[i].createOutStr(charrTemp);
        strcat(arrch, charrTemp);
        memset(charrTemp, 0, 200 * sizeof(char));
        sprintf(charrTemp, ", Length of pulse= %.10e", pulse_length);
        strcat(arrch, charrTemp);
        //strcat(arrch, "\n");
        strcat(arrch, "\n");
        size_t elements_written = fwrite(arrch, sizeof(char), strlen(arrch),wb_file);

    }  
    fclose(wb_file);
}



