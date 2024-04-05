#include "DrawImg.h"
#include <math.h>
#include <stdio.h>
#include <array>
#include <iostream>
#include <string>

#include <GL/glut.h>
#include <GL/gl.h>
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>
#include <vector>
#include <cstdlib> // For random value generation
#include <ctime>   // For seeding the random number generator
#include "npy.hpp"
#include <algorithm> 

#include <chrono>
#include "fileInput.h"
#include "wchar.h"



extern int IROWS = 0;
extern int ICOLS = 0;
extern  std::vector<std::vector<int>> ivctOut = std::vector<std::vector<int>>(1, std::vector<int>(1, 0));

void display() {
    //glClearColor(0.0, 0.0, 0.0, 0.0); // Background color (black)
    glClearColor(1.0, 1.0, 1.0, 0.0); // Background color (white)

    glClear(GL_COLOR_BUFFER_BIT);

    // Set up the view
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, ICOLS, 0, IROWS);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    int maxData = ivctOut[0][0]; // Initialize maxData with the first element

    int minData = ivctOut[0][0]; // Initialize maxData with the first element


    auto maxElement = std::max_element(ivctOut.begin(), ivctOut.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        return *std::max_element(a.begin(), a.end()) < *std::max_element(b.begin(), b.end());
        });

    maxData = *std::max_element(maxElement->begin(), maxElement->end());

    auto minElement = std::min_element(ivctOut.begin(), ivctOut.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        return *std::min_element(a.begin(), a.end()) < *std::min_element(b.begin(), b.end());
        });

    minData = *std::min_element(minElement->begin(), minElement->end());

    // Render the piarrOut array as an image here
    glBegin(GL_POINTS);
    for (int i = 0; i < IROWS; i++) {
        for (int j = 0; j < ICOLS; j++)
        {
            float grayscale = static_cast<float>(ivctOut[i][j]);// / 255.0;
            // Calculate RGB values based on linear mapping
            float red = (grayscale - minData) / (maxData - minData);
            float green = 1.0 - red;  // Set to 0 for simplicity
            float blue = 0.25 * (1.0 - red);

            glColor3f(red, green, blue); // Grayscale color
            glVertex2f(j, i); // Draw a point for each pixel
        }
    }
    glEnd();

    glFlush();
}
//--------------------------------------------------------------------------------------

void saveImage(const char* filename)
{
    ilInit();
    ilutRenderer(ILUT_OPENGL);
    ilEnable(IL_FILE_OVERWRITE);

    ILuint imageID = ilGenImage();
    ilBindImage(imageID);

    ILenum format = IL_LUMINANCE;
    ILenum type = IL_UNSIGNED_BYTE;

    std::vector<ILubyte> pixelData(IROWS * ICOLS);

    for (int i = 0; i < IROWS; i++) {
        for (int j = 0; j < ICOLS; j++) {
            pixelData[i * ICOLS + j] = static_cast<ILubyte>(ivctOut[i][j]);
        }
    }

    ilTexImage(ICOLS, IROWS, 1, 1, format, type, pixelData.data());
    ilSave(IL_PNG, filename);

    ilDeleteImages(1, &imageID);
}

void createImg(int argc, char** argv, int * piarrImOut, const int IRows, const int ICols,const char* filename)
{
    ICOLS = ICols;
    IROWS = IRows;
    int* pi = new int[ICOLS];

    int num = IROWS / 2;

    for (int i = 0; i < num; ++i)
    {
        memcpy(pi, &piarrImOut[i * ICOLS], ICOLS * sizeof(int));
        memcpy(&piarrImOut[i * ICOLS], &piarrImOut[(IROWS - 1 - i) * ICOLS], ICOLS * sizeof(int));
        memcpy(&piarrImOut[(IROWS - 1 - i) * ICOLS], pi, ICOLS * sizeof(int));
    }
    delete []pi;
    int imax = *std::max_element(piarrImOut, piarrImOut + ICOLS * IROWS);
    int imin = *std::min_element(piarrImOut, piarrImOut + ICOLS * IROWS);
    float coeff = 255. / (double(imax));
    ivctOut = std::vector<std::vector<int>>(IROWS, std::vector<int>(ICOLS, 0)); // Initialize with your data
    for (int i = 0; i < IROWS; ++i)
        for (int j = 0; j < ICOLS; ++j)
        {
            ivctOut[i][j] = (int)(coeff * piarrImOut[i * ICOLS + j]);
            ivctOut[i][j] = (int)piarrImOut[i * ICOLS + j];

        }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(ICOLS, IROWS);
    glutCreateWindow(filename);

    glutDisplayFunc(display);
    // After displaying the image, save it as filename 
    saveImage(filename);
    
    // Set up other GLUT callbacks as needed (e.g., keyboard input)

    glutMainLoop();
    
    
    return;
}

void createImg_(int argc, char** argv, std::vector<int>& vctOut1D
    , const int IRows, const int ICols, const char* filename)
{
    ICOLS = ICols;
    IROWS = IRows;
    ivctOut = std::vector<std::vector<int>>(IROWS, std::vector<int>(ICOLS, 0));
    int imax = fabs(vctOut1D[0]);

    for (int i = 0; i < IROWS; ++i)
    {
        for (int j = 0; j < ICOLS; ++j)
        {
            int t = vctOut1D[i * ICOLS + j];
            ivctOut[i][j] = t;
            if (fabs(t) > imax)
            {
                imax = t;
            }
        }
    }
    /*std::swap(ivctOut[2], ivctOut[3]);
    int* pi = new int[ICOLS];*/

    int num = IROWS / 2;

    for (int i = 0; i < num; ++i)
    {
        std::swap(ivctOut[i], ivctOut[IROWS - 1 - i]);
        /*memcpy(pi, &piarrImOut[i * ICOLS], ICOLS * sizeof(int));
        memcpy(&piarrImOut[i * ICOLS], &piarrImOut[(IROWS - 1 - i) * ICOLS], ICOLS * sizeof(int));
        memcpy(&piarrImOut[(IROWS - 1 - i) * ICOLS], pi, ICOLS * sizeof(int));*/
    }
    //delete[]pi;
    /*int imax = *std::max_element(piarrImOut, piarrImOut + ICOLS * IROWS);
    int imin = *std::min_element(piarrImOut, piarrImOut + ICOLS * IROWS);*/
    float coeff = 255. / (double(imax));
    // Initialize with your data
    for (int i = 0; i < IROWS; ++i)
        for (int j = 0; j < ICOLS; ++j)
        {
            ivctOut[i][j] = (int)(coeff * ivctOut[i][j]);
            //ivctOut[i][j] = (int)piarrImOut[i * ICOLS + j];

        }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(ICOLS, IROWS);
    glutCreateWindow(filename);

    glutDisplayFunc(display);
    // After displaying the image, save it as filename 
    saveImage(filename);

    // Set up other GLUT callbacks as needed (e.g., keyboard input)

    glutMainLoop();


    return;
}
//--------------------------------------------------------------
//template <typename T>
//void createImg__(int argc, char** argv, std::vector<T>& vctOut1D
//    , const int IRows, const int ICols, const char* filename)
//{
//    ICOLS = ICols;
//    IROWS = IRows;
//    ivctOut = std::vector<std::vector<int>>(IROWS, std::vector<int>(ICOLS, 0));
//    int imax = fabs(vctOut1D[0]);
//
//    for (int i = 0; i < IROWS; ++i)
//    {
//        for (int j = 0; j < ICOLS; ++j)
//        {
//            int t = vctOut1D[i * ICOLS + j];
//            ivctOut[i][j] = t;
//            if (fabs(t) > imax)
//            {
//                imax = t;
//            }
//        }
//    }
//    /*std::swap(ivctOut[2], ivctOut[3]);
//    int* pi = new int[ICOLS];*/
//
//    int num = IROWS / 2;
//
//    for (int i = 0; i < num; ++i)
//    {
//        std::swap(ivctOut[i], ivctOut[IROWS - 1 - i]);
//        /*memcpy(pi, &piarrImOut[i * ICOLS], ICOLS * sizeof(int));
//        memcpy(&piarrImOut[i * ICOLS], &piarrImOut[(IROWS - 1 - i) * ICOLS], ICOLS * sizeof(int));
//        memcpy(&piarrImOut[(IROWS - 1 - i) * ICOLS], pi, ICOLS * sizeof(int));*/
//    }
//    //delete[]pi;
//    /*int imax = *std::max_element(piarrImOut, piarrImOut + ICOLS * IROWS);
//    int imin = *std::min_element(piarrImOut, piarrImOut + ICOLS * IROWS);*/
//    float coeff = 255. / (double(imax));
//    // Initialize with your data
//    for (int i = 0; i < IROWS; ++i)
//        for (int j = 0; j < ICOLS; ++j)
//        {
//            ivctOut[i][j] = (int)(coeff * ivctOut[i][j]);
//           // ivctOut[i][j] = (int)piarrImOut[i * ICOLS + j];
//
//        }
//
//    glutInit(&argc, argv);
//    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
//    glutInitWindowSize(ICOLS, IROWS);
//    glutCreateWindow(filename);
//
//    glutDisplayFunc(display);
//    // After displaying the image, save it as filename 
//    saveImage(filename);
//
//    // Set up other GLUT callbacks as needed (e.g., keyboard input)
//
//    glutMainLoop();
//
//
//    return;
//}
//---------------------------------------------
void createImg_(int argc, char** argv, std::vector<float>& vctOut1D
    , const int IRows, const int ICols, const char* filename)
{
    ICOLS = ICols;
    IROWS = IRows;
    ivctOut = std::vector<std::vector<int>>(IROWS, std::vector<int>(ICOLS, 0));
    int imax = fabs(vctOut1D[0]);

    for (int i = 0; i < IROWS; ++i)
    {
        for (int j = 0; j < ICOLS; ++j)
        {
            int t = vctOut1D[i * ICOLS + j];
            ivctOut[i][j] = t;
            if (fabs(t) > imax)
            {
                imax = t;
            }
        }
    }
    /*std::swap(ivctOut[2], ivctOut[3]);
    int* pi = new int[ICOLS];*/

    int num = IROWS / 2;

    for (int i = 0; i < num; ++i)
    {
        std::swap(ivctOut[i], ivctOut[IROWS - 1 - i]);
        /*memcpy(pi, &piarrImOut[i * ICOLS], ICOLS * sizeof(int));
        memcpy(&piarrImOut[i * ICOLS], &piarrImOut[(IROWS - 1 - i) * ICOLS], ICOLS * sizeof(int));
        memcpy(&piarrImOut[(IROWS - 1 - i) * ICOLS], pi, ICOLS * sizeof(int));*/
    }
    //delete[]pi;
    /*int imax = *std::max_element(piarrImOut, piarrImOut + ICOLS * IROWS);
    int imin = *std::min_element(piarrImOut, piarrImOut + ICOLS * IROWS);*/
    float coeff = 255. / (double(imax));
    // Initialize with your data
    for (int i = 0; i < IROWS; ++i)
        for (int j = 0; j < ICOLS; ++j)
        {
            ivctOut[i][j] = (int)(coeff * ivctOut[i][j]);
            //ivctOut[i][j] = (int)piarrImOut[i * ICOLS + j];

        }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(ICOLS, IROWS);
    glutCreateWindow(filename);

    glutDisplayFunc(display);
    // After displaying the image, save it as filename 
    saveImage(filename);

    // Set up other GLUT callbacks as needed (e.g., keyboard input)

    glutMainLoop();


    return;
}


