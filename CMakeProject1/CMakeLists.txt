cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

# Specify the CUDA Toolkit version
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin/nvcc.exe")

# Specify the CUDA architecture
set(CUDA_ARCHITECTURE "75") # Change this based on your GPU architecture

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
find_package(CUDA REQUIRED)

# Add the executable
add_executable(CudaProject main.cu)

# Link CUDA libraries
target_link_libraries(CudaProject ${CUDA_LIBRARIES})

# Specify include directories
include_directories(${CUDA_INCLUDE_DIRS})
