/*
 * GMEMcommonCUDA.h
 *
 */

#ifndef GMEMCOMMONCUDA_H_
#define GMEMCOMMONCUDA_H_

#include "cuda.h"
#include "../constants.h"

static const int MAX_REF_POINTS=5000;//we need to predefined this in order to store reference points as constant memory. Total memory needed is MAX_QUERY_POINTS*3*4 bytes. It can not be more than 5400!!!

#ifndef CUDA_MAX_SIZE_CONST //to protect agains teh same constant define in other places in the code
#define CUDA_MAX_SIZE_CONST
static const int MAX_THREADS=1024;//For Quadro4800->256;//to make sure we don't run out of registers
static const int MAX_THREADS_CUDA=1024;//For Quadro4800->512;//certain kernels benefit from maximum number of threads
static const int MAX_BLOCKS=65535;
#endif

static const int TILE_DIM = 6*maxGaussiansPerVoxel;//transposing elements.. It has to be a multiple of maxGaussiansPerVoxel and below 512
static const int BLOCK_ROWS = 16;
//static const int MAX_TEXTURE1D_MEM=128;//in MBytes

__device__ static const double PI_CUDA=3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651;

__device__ static const double PI_CUDAf=3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651f;

//for some reason VS chokes in here and I need to redefine the value or use __constant__ and upload value to device memory once
#if defined(_WIN32) || defined(_WIN64) 
	__device__ const double minPi_kForDead_CUDA=1e-5;
#else
	__device__ static const double minPi_kForDead_CUDA=minPi_kForDead;
#endif

//texture<float> queryTexture;

#endif /* GMEMCOMMONCUDA_H_ */
