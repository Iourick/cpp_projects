#pragma once
#include "OutChunkHeader.h"
class COutChunkHeader;
class COutChunk
{
public:
	~COutChunk();
	COutChunk();
	COutChunk(const  COutChunk& R);
	COutChunk& operator=(const COutChunk& R);
	COutChunk(const COutChunkHeader OutChunkHeader, const float* pfdmtOut);
	//--------------------------------------

	COutChunkHeader m_OutChunkHeader;
	float* m_pfdmtOut;
};




