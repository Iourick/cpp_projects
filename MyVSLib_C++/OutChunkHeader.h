#pragma once
#include <vector>
class COutChunkHeader
{
public:
	~COutChunkHeader() ;
	COutChunkHeader();
	COutChunkHeader(const  COutChunkHeader& R);
	COutChunkHeader& operator=(const COutChunkHeader& R);
	COutChunkHeader(
		const int nrows
		, const int ncols
		, const int nSucessRow
		, const int nSucessCol
		, const int width
		, const float SNR
		, const float 	coherentDedisp
		, const int numBlock
		, const int numChunk
	);
	// quant rows of output fdmt image:
	int m_nrows;
	// quant cols of output fdmt image:
	int m_ncols;
	// number of suceeded  row
	int m_nSucessRow;
	// number of suceeded  col
	int m_nSucessCol;
	// SNR:
	float m_SNR;
	//suceeded coherent dedispersion value
	float m_coherentDedisp;
	// number of block
	int m_numBlock;
	// number of chunk
	int m_numChunk;
	// succeeded window width
	int m_wnd_width;

	void createOutStr(char* pstr);

	static bool read_outputlogfile_line(const char* pstrPassLog
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
	);

	static void writeReport(const char* chstrOutput, std::vector<COutChunkHeader>* pvctSuccessHeaders
		, const float pulse_length);

};

