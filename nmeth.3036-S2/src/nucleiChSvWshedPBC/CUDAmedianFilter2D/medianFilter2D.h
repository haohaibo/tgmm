/*
 * Copyright (C) 2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  medianFilter2D.h
 *
 *  Created on: January 17th, 2013
 *      Author: Fernando Amat
 *
 * \brief Code to calculate 2D median filter in CUDA using templates and different window sizes
 * \Note: this code can be easily used for any non-linear processing in a neighborhhod block around each pixel with a simple modoifcation. Analogous to blockproc by Matlab
 */

#ifndef __MEDIAN_FILTER_2D_CUDA_H__
#define __MEDIAN_FILTER_2D_CUDA_H__


//define constants

#ifndef CUDA_CONSTANTS_FA
#define CUDA_CONSTANTS_FA
static const int MAX_THREADS_CUDA = 1024; //adjust it for your GPU. This is correct for a 2.0 architecture
static const int MAX_BLOCKS_CUDA = 65535;
static const int BLOCK_SIDE = 32; //we load squares into share memory. 32*32 = 1024, which is the maximum number of threads per block for CUDA arch 2.0. 
#endif

#ifndef DIMS_IMAGE_SLICE
#define DIMS_IMAGE_SLICE
static const int dimsImageSlice = 2;//so thing can be set at co0mpile time

#endif



/*
	\brief Main function to calculaye median filter with CUDA
	
	\param[in/out]		im			pointer to image in host. It is overwritten by the median result
	\param[in]			imDim		array of length dimsImageSlice indicating the image dimensions. imDim[0] is the fastest running index in memory.
	\param[in]			radius		radius of the median filter. Median filter window size is 2*radius +1. So a 3x3 median filter has radius = 1
	\param[in]			devCUDA		in case you have multiple GPU in the same computer
*/

template<class imgType>
int medianFilterCUDA(imgType* im,int* imDim,int radius,int devCUDA);


/*
	\brief apply medianFilter2D to a 3D stack slice by slice (or an RGB stack)
	
*/

template<class imgType>
int medianFilterCUDASliceBySlice(imgType* im,int* imDim,int radius,int devCUDA);




#endif //__CONVOLUTION_3D_FFT_H__