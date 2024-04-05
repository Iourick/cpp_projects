#include "read_and_write_log.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>


#ifdef _WIN32 // For Windows
#include <Windows.h>
#include <Lmcons.h> // Required for UNLEN constant
#else // For Linux
#include <unistd.h>
#include <limits.h>
#endif
#define _CRT_SECURE_NO_WARNINGS
using namespace std;


int fncReadLog(const char* filename, int* lengthOfChunk, int* quantChunks, int* arrChunks)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Unable to open the file " << filename << " for reading!" << std::endl;
        return -1; // Return -1 to indicate failure in opening the file
    }

    std::string line;
    std::vector<std::string> lines;

    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Parsing the required information from the lines
    for (const auto& line : lines)
    {
        if (line.find("Length of Chunk: ") != std::string::npos)
        {
            *lengthOfChunk = std::stoi(line.substr(16)); // Extracting the length of chunk
        }
        else
        {
            if (line.find("Quant of Successful Chunks: ") != std::string::npos)
            {
                *quantChunks = std::stoi(line.substr(27)); // Extracting the length of chunk
            }
            else
            {
                if (line.find("Numbers of Successful Chunks: [") != std::string::npos)
                {
                    std::stringstream ss(line.substr(31, line.length() - 33));
                    int value;
                    int index = 0;
                    while (ss >> value && index < *lengthOfChunk)
                    {
                        arrChunks[index] = value;
                        ++index;
                        if (ss.peek() == ',') ss.ignore();
                    }
                }
            }


        }
    }

    file.close();


    return 0; // Return 0 to indicate success
}


int fncReadLog_(const char* filename, char* passDatafile, int* lengthOfChunk, int* quantChunks
    , int* arrChunks, float* arrCohD)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Unable to open the file " << filename << " for reading!" << std::endl;
        return -1; // Return -1 to indicate failure in opening the file
    }

    std::string line;
    std::vector<std::string> lines;

    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    //--

    std::string fullString = lines[0];

    std::string searchString = "Pass to Data File: ";
    size_t startPos = fullString.find(searchString);

    if (startPos != std::string::npos)
    {
        std::string extractedString = fullString.substr(startPos + searchString.length());      
        
        strcpy(passDatafile, extractedString.c_str());       
    }
    else
    {
        std::cout << "Substring not found in the string." << std::endl;
    }
    //--

    // Parsing the required information from the lines
    for (const auto& line : lines)
    {
        if (line.find("Length of Chunk: ") != std::string::npos)
        {
            *lengthOfChunk = std::stoi(line.substr(16)); // Extracting the length of chunk
        }
        else
        {
            if (line.find("Quant of Successful Chunks: ") != std::string::npos)
            {
                *quantChunks = std::stoi(line.substr(27)); // Extracting the length of chunk
            }
            else
            {
                if (line.find("Numbers of Successful Chunks: [") != std::string::npos)
                {
                    std::stringstream ss(line.substr(31, line.length() - 1));
                    int value;
                    int index = 0;
                    while (ss >> value && index < *lengthOfChunk)
                    {
                        arrChunks[index] = value;
                        ++index;
                        if (ss.peek() == ',') ss.ignore();
                    }
                }
                else
                {
                    if (line.find("Successfull coherentD Array: [") != std::string::npos)
                    {
                        //std::stringstream ss(line.substr(30, line.length() - 33));
                        std::stringstream ss(line.substr(30, line.length()-1));
                        float value;
                        int index = 0;
                        while (ss >> value && index < *lengthOfChunk)
                        {
                            arrCohD[index] = value;
                            ++index;
                            if (ss.peek() == ',') ss.ignore();
                        }
                    }
                }
            }


        }
    }

    file.close();
    
    return 0; // Return 0 to indicate success
}

void fncWriteLog(const char* filename, const char* dataFilePass, const char* strProjectName, int lengthOfChunk
    , int quantOfChunk, const int* successfulChunks, int runningTime)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << "Pass to Data File: " << dataFilePass << "\n";
        outputFile << "Project Name: " << strProjectName << "\n";
        outputFile << "Length of Chunk: " << lengthOfChunk << "\n";
        outputFile << "Quant of Successful Chunks: " << quantOfChunk << "\n";
        outputFile << "Numbers of Successful Chunks: [";
        for (int i = 0; i < quantOfChunk; ++i) {
            outputFile << successfulChunks[i];
            if (i != lengthOfChunk - 1) {
                outputFile << ", ";
            }
        }
        outputFile << "]\n";
        outputFile << "Running Time (seconds): " << runningTime << "\n";
        outputFile.close();
        std::cout << "Information has been written to " << filename << " successfully." << std::endl;
    }
    else {
        std::cerr << "Unable to open the file " << filename << " for writing!" << std::endl;
    }
}


void fncWriteLog_(const char* filename, const char* dataFilePass, const char* strProjectName, int lengthOfChunk
    , int quantOfChunk, const int* successfulChunks, const float* arr_coh_d, int runningTime)
{
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << "Pass to Data File: " << dataFilePass << "\n";
        outputFile << "Project Name: " << strProjectName << "\n";
        outputFile << "Length of Chunk: " << lengthOfChunk << "\n";
        outputFile << "Quant of Successful Chunks: " << quantOfChunk << "\n";
        outputFile << "Numbers of Successful Chunks: [";
        for (int i = 0; i < quantOfChunk; ++i) {
            outputFile << successfulChunks[i];
            if (i != lengthOfChunk - 1) {
                outputFile << ", ";
            }
        }
        outputFile << "]\n";
        outputFile << "Successfull coherentD Array: [";
        for (int i = 0; i < quantOfChunk; ++i) {
            outputFile << arr_coh_d[i];
            if (i != lengthOfChunk - 1) {
                outputFile << ", ";
            }
        }
        outputFile << "]\n";
        outputFile << "Running Time (seconds): " << runningTime << "\n";
        outputFile.close();
        std::cout << "Information has been written to " << filename << " successfully." << std::endl;
    }
    else {
        std::cerr << "Unable to open the file " << filename << " for writing!" << std::endl;
    }
}
