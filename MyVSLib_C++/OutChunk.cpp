#include "OutChunk.h"
#include <iostream>
COutChunk::~COutChunk()
{
	if (m_pfdmtOut)
	{
		free(m_pfdmtOut);
	}
	m_pfdmtOut = NULL;	
}

COutChunk::COutChunk()
{
	m_OutChunkHeader = COutChunkHeader();
	m_pfdmtOut = NULL;	
}
//-----------------------------------------------------------

COutChunk::COutChunk(const  COutChunk& R)
{
	m_OutChunkHeader = R.m_OutChunkHeader;
	
	if (m_pfdmtOut)
	{
		free(m_pfdmtOut);
		m_pfdmtOut = NULL;
	}
	m_pfdmtOut = (float*)malloc(m_OutChunkHeader.m_ncols * m_OutChunkHeader.m_nrows * sizeof(float));
	memcpy(m_pfdmtOut, R.m_pfdmtOut, m_OutChunkHeader.m_ncols * m_OutChunkHeader.m_nrows * sizeof(float));
}
//-------------------------------------------------------------------

COutChunk& COutChunk::operator=(const COutChunk& R)
{
	if (this == &R)
	{
		return *this;
	}

	m_OutChunkHeader = R.m_OutChunkHeader;

	if (m_pfdmtOut)
	{
		free(m_pfdmtOut);
		m_pfdmtOut = NULL;
	}
	m_pfdmtOut = (float*)malloc(m_OutChunkHeader.m_ncols * m_OutChunkHeader.m_nrows * sizeof(float));
	memcpy(m_pfdmtOut, R.m_pfdmtOut, m_OutChunkHeader.m_ncols * m_OutChunkHeader.m_nrows * sizeof(float));
	return *this;
}
//------------------------------------------------------------------
COutChunk::COutChunk(const COutChunkHeader OutChunkHeader,const float* pfdmtOut)

{
	m_OutChunkHeader = OutChunkHeader;

	if (pfdmtOut)
	{
		if (m_pfdmtOut)
		{
			//free(m_pfdmtOut);
			m_pfdmtOut = NULL;
		}
		if (!(m_pfdmtOut = (float*)malloc(m_OutChunkHeader.m_ncols * m_OutChunkHeader.m_nrows * sizeof(float))))
		{
			std::cout << "Can't allocate memory for m_pfdmtOut in OutPutChunk.cpp" << std::endl;
		}
		if (!memcpy(m_pfdmtOut, pfdmtOut, m_OutChunkHeader.m_ncols * m_OutChunkHeader.m_nrows * sizeof(float)))
		{
			std::cout << "Can't memcpy pfdmtOut in OutPutChunk.cpp" << std::endl;
		}
	}

}
