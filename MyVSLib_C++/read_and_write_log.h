#pragma once
int fncReadLog(const char* filename, int* lengthOfChunk, int* quantChunks, int* arrChunks);

void fncWriteLog(const char* filename, const char* projectName, const char* strProjectName, int lengthOfChunk
    , int quantOfChunk, const int* successfulChunks, int runningTime);

void fncWriteLog_(const char* filename, const char* dataFilePass, const char* strProjectName, int lengthOfChunk
    , int quantOfChunk, const int* successfulChunks, const float* arr_coh_d, int runningTime);

int fncReadLog_(const char* filename, char* passDatafile, int* lengthOfChunk, int* quantChunks
    , int* arrChunks, float* arrCohD);
