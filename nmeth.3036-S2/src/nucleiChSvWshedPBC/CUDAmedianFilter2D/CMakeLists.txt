#CMake file to build Tracking_GMM project

#define important variables for the rest of the project
set(SETBYUSER_CUDA_ARCH sm_20 CACHE STRING "CUDA architecture") #for Quadro FX4800 sm13;fir Tesla sm_20

# refer to the root source directory of the project as ${TGMM_SOURCE_DIR} and
# to the root binary directory of the project as ${TGMM_BINARY_DIR}. 
#The FindCUDA script is distributed since version 2.8

cmake_minimum_required (VERSION 2.8)
project (MedianFilter2D_project)



#
#Set important flags
#

# Call  "cmake-gui -DCMAKE_BUILD_TYPE=Release .." on the command line to specify a particular build, take their definition
# and dump it in the cache along with proper documentation, otherwise set CMAKE_BUILD_TYPE
# to Debug prior to calling PROJECT()
#
IF( CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ELSE() #Debug is the default
	SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

MESSAGE("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

#to locate scripts
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake") 

#
# Find necessary libraries
#
#Cuda
#Any device debugging symbols (-G) generate a kernel launch failure in thurst::sort_by_key
SET(CUDA_NVCC_FLAGS_DEBUG -DTHRUST_DEBUG;-G;-O0;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments") #set before FIND_PACKAGE(CUDA) in order to avoid FORCE to show them in GUI. So user can modify them
SET(CUDA_NVCC_FLAGS_RELEASE -O3;-arch=${SETBYUSER_CUDA_ARCH} CACHE STRING "Semi-colon delimit multiple arguments")
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_VERBOSE_BUILD ON FORCE)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS}) 
#lcudart and lcuda  are already added as variables with find package
#SET(LIBS_CUDA_CUSPARSE "cusparse" CACHE STRING "")#add CUDA libraries ignored in CUDA.cmake


#
# submodules: add source and header files from subfolders
#
file(GLOB CUDA_HDRS *.h)
file(GLOB CUDA_CUDA_SRCS *.cu)

file(GLOB TICTOC_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/external/Nathan/*.h)
file(GLOB TICTOC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/external/Nathan/*.c)


#TODO SET THE FLAGS FOR DEBUG VERSION
#set(CUOPTS -fPIC)
CUDA_COMPILE(CUDA_CUDA_OBJ STATIC ${CUDA_CUDA_SRCS} OPTIONS ${CUOPTS})

#compile executable
IF(NOT TARGET MedianFilter2D)
#add_definitions( -fPIC )
add_executable(MedianFilter2D mainTest.cpp ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} ${TICTOC_HDRS} ${TICTOC_SRCS})

target_link_libraries(MedianFilter2D ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
ENDIF()



#generate also library
IF(NOT TARGET MedianFilter2D_CUDAlib)
add_library(MedianFilter2D_CUDAlib ${CUDA_SRCS} ${CUDA_CUDA_OBJ} ${CUDA_HDRS} ${CUDA_CUDA_SRCS} ${TICTOC_HDRS} ${TICTOC_SRCS})
#link libraries
TARGET_LINK_LIBRARIES(MedianFilter2D_CUDAlib ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
ENDIF()