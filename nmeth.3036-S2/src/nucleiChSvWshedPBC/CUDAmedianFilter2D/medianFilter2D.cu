/*
* Copyright (C) 2013 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat 
*  medianFilter2D.cu
*
*  Created on: January 17th, 2013
*      Author: Fernando Amat
*
* \brief Code to calculate 2D median filter in CUDA using templates and different window sizes
*
*/

#include "medianFilter2D.h"
#include "book.h"
#include "cuda.h"
#include <iostream>
#include <math.h>



__constant__ int imDimCUDA[dimsImageSlice];//image dimensions



//=====================================================================================
template<class imgType, int radius>
__global__ void __launch_bounds__(BLOCK_SIDE*BLOCK_SIDE) medianFilterCUDAkernel(imgType* imCUDAin, imgType* imCUDAout, unsigned int imSize)
{

	const int radiusSize = ( 1 + 2 * radius) * ( 1 + 2 * radius);
	//shared memory to copy global memory
	__shared__ imgType blockNeigh [BLOCK_SIDE * BLOCK_SIDE];//stores values for a whole block
	imgType imgNeigh[ radiusSize ];//store values for each thread. This is the reason why radius is a template parameter and not a function input variable. Here we have a chance that everything fits in register memory for small radiuses
	
	
	int offset_x = blockIdx.x * (BLOCK_SIDE - 2* radius) - radius + threadIdx.x;//upper left corner of the image to start loading into share memory (counting overlap to accomodate radius)
	int offset_y = blockIdx.y * (BLOCK_SIDE - 2* radius) - radius + threadIdx.y;//upper left corner of the image to start loading into share memory (counting overlap to accomodate radius)
	
	int tid = threadIdx.y * BLOCK_SIDE + threadIdx.x;

	//each thread loads one pixel into share memory (colescent access)
	int pos;
	if( offset_x < 0 || offset_y < 0 || offset_x >= imDimCUDA[0] || offset_y >= imDimCUDA[1] )//out of bounds
	{
		pos = -1;
		blockNeigh[ tid ] = 0;//for now we assume zeros outside image boundaries
	}else{
		pos = offset_x + offset_y  * imDimCUDA[0];
		blockNeigh[ tid ] = imCUDAin[pos];
	}

	__syncthreads();

	if( threadIdx.x < radius || threadIdx.x >= BLOCK_SIDE-radius || threadIdx.y < radius || threadIdx.y >= BLOCK_SIDE-radius)
		return;//these threads are not needed (kind of a waste, but it is OK);

	
	//operate on block: this part could be substituted by any other operation in a blokc if we want to apply a different filter than median		
	int pp, count = 0;
	for( int ii = -radius; ii <= radius; ii++)
	{
		pp = threadIdx.x -radius + BLOCK_SIDE * ( threadIdx.y + ii );//initial position for jj for loop		

		for( int jj = -radius; jj <= radius; jj++)
		{				
			imgNeigh[count++] = blockNeigh[pp++];
		}
	}

	//selection algorithm to find the k-th smallest number (k = (radiusSize - 1) /2 (http://en.wikipedia.org/wiki/Selection_algorithm)
	imgType temp;
	for ( int ii=0; ii < (1 + radiusSize) /2; ii++)
	{
		// Find position of minimum element
		pp = ii;//minIndex
		for ( int jj = ii+1; jj < radiusSize; jj++)
		{
			if (imgNeigh[jj] < imgNeigh[pp])
			{
				pp = jj;
			}
		}
		temp = imgNeigh[pp];
		imgNeigh[pp] = imgNeigh[ii];
		imgNeigh[ii] = temp;
	}	

	if( pos>=0 && pos< imSize )
		imCUDAout[ pos ] = imgNeigh[ (radiusSize - 1) /2 ];
};



//===========================================================================

template<class imgType>
int medianFilterCUDA(imgType* im,int* imDim,int radius,int devCUDA)
{
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	if( radius > (int)(floor( (BLOCK_SIDE -1) / 2.0f) )  || 2 * radius >= BLOCK_SIDE)
	{
		std::cout<<"ERROR: at medianFilterCUDA: code is not ready for such a large radius. Maximum radius allowed is "<<(int)(floor( (BLOCK_SIDE - 1) / 2.0f) )<<std::endl;
		return 2;
	}


	imgType* imCUDAinput = NULL;
	imgType* imCUDAoutput = NULL;


	int imSize = imDim[0];
	for( int ii = 1; ii < dimsImageSlice; ii++)
		imSize *= imDim[ii];

	//allocate memory in CUDA (input and output)
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDAinput), imSize * sizeof(imgType) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDAoutput), imSize * sizeof(imgType) ) );

	//transfer input: image and image dimensions
	HANDLE_ERROR(cudaMemcpy(imCUDAinput, im, imSize * sizeof(imgType), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(imDimCUDA,imDim, dimsImageSlice * sizeof(int)));//constant memory

	//run kernel	
	dim3 threads( BLOCK_SIDE, BLOCK_SIDE );
	int numBlocks[dimsImageSlice];
	for (int ii = 0 ; ii< dimsImageSlice; ii++)
		numBlocks[ii] = (int) (ceil( (float)(imDim[ii] + radius ) / (float)(BLOCK_SIDE - 2 * radius) ) );
	dim3 blocks(numBlocks[0], numBlocks[1]);//enough to cover all the image

	switch(radius)
	{
	case 0:
		//do nothing
		break;
	case 1:
		medianFilterCUDAkernel<imgType, 1> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 2:
		medianFilterCUDAkernel<imgType, 2> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 3:
		medianFilterCUDAkernel<imgType, 3> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 4:
		medianFilterCUDAkernel<imgType, 4> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 5:
		medianFilterCUDAkernel<imgType, 5> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 6:
		medianFilterCUDAkernel<imgType, 6> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 7:
		medianFilterCUDAkernel<imgType, 7> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 8:
		medianFilterCUDAkernel<imgType, 8> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 9:
		medianFilterCUDAkernel<imgType, 9> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 10:
		medianFilterCUDAkernel<imgType, 10> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	case 11:
		medianFilterCUDAkernel<imgType, 11> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
		break;
	default:
		std::cout<<"ERROR: at medianFilterCUDA: code is not ready for such a large radius." <<std::endl;//If I need it at any point, I could extend this up to (int)(floor( (BLOCK_SIDE -1) / 2.0f) )
		return 4;
	}
	//copy result to host
	HANDLE_ERROR(cudaMemcpy(im, imCUDAoutput, imSize * sizeof(imgType), cudaMemcpyDeviceToHost));

	//deallocate memory
	HANDLE_ERROR( cudaFree( imCUDAinput ) );
	HANDLE_ERROR( cudaFree( imCUDAoutput ) );

	return 0;
}

//declare all the possible types so template compiles properly
template int medianFilterCUDA<unsigned char>(unsigned char* im,int* imDim,int radius,int devCUDA);
template int medianFilterCUDA<unsigned short int>(unsigned short int* im,int* imDim,int radius,int devCUDA);
template int medianFilterCUDA<float>(float* im,int* imDim,int radius,int devCUDA);


//===========================================================================

template<class imgType>
int medianFilterCUDASliceBySlice(imgType* im,int* imDim,int radius,int devCUDA)
{
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	if( radius > (int)(floor( (BLOCK_SIDE -1) / 2.0f) )  || 2 * radius >= BLOCK_SIDE)
	{
		std::cout<<"ERROR: at medianFilterCUDA: code is not ready for such a large radius. Maximum radius allowed is "<<(int)(floor( (BLOCK_SIDE - 1) / 2.0f) )<<std::endl;
		return 2;
	}


	imgType* imCUDAinput = NULL;
	imgType* imCUDAoutput = NULL;


	int imSize = imDim[0];
	for( int ii = 1; ii < dimsImageSlice; ii++)
		imSize *= imDim[ii];

	//allocate memory in CUDA (input and output)
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDAinput), imSize * sizeof(imgType) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDAoutput), imSize * sizeof(imgType) ) );

	//kernel parameters
	dim3 threads( BLOCK_SIDE, BLOCK_SIDE );
	int numBlocks[dimsImageSlice];
	for (int ii = 0 ; ii< dimsImageSlice; ii++)
		numBlocks[ii] = (int) (ceil( (float)(imDim[ii] + radius ) / (float)(BLOCK_SIDE - 2 * radius) ) );
	dim3 blocks(numBlocks[0], numBlocks[1]);//enough to cover all the image

	//perform median filter slice by slice
	for( int slice = 0; slice < imDim[dimsImageSlice ]; slice++)
	{

		//transfer input: image and image dimensions
		HANDLE_ERROR(cudaMemcpy(imCUDAinput, im, imSize * sizeof(imgType), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpyToSymbol(imDimCUDA,imDim, dimsImageSlice * sizeof(int)));//constant memory

		//run kernel			
		switch(radius)
		{
		case 0:
			//do nothing
			break;
		case 1:
			medianFilterCUDAkernel<imgType, 1> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 2:
			medianFilterCUDAkernel<imgType, 2> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 3:
			medianFilterCUDAkernel<imgType, 3> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 4:
			medianFilterCUDAkernel<imgType, 4> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 5:
			medianFilterCUDAkernel<imgType, 5> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 6:
			medianFilterCUDAkernel<imgType, 6> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 7:
			medianFilterCUDAkernel<imgType, 7> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 8:
			medianFilterCUDAkernel<imgType, 8> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 9:
			medianFilterCUDAkernel<imgType, 9> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 10:
			medianFilterCUDAkernel<imgType, 10> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		case 11:
			medianFilterCUDAkernel<imgType, 11> <<<blocks, threads>>>(imCUDAinput, imCUDAoutput, imSize);HANDLE_ERROR_KERNEL;
			break;
		default:
			std::cout<<"ERROR: at medianFilterCUDA: code is not ready for such a large radius." <<std::endl;//If I need it at any point, I could extend this up to (int)(floor( (BLOCK_SIDE -1) / 2.0f) )
			return 4;
		}
		//copy result to host
		HANDLE_ERROR(cudaMemcpy(im, imCUDAoutput, imSize * sizeof(imgType), cudaMemcpyDeviceToHost));

		im += imSize;//increment pointer to next slice
	}


	//deallocate memory
	HANDLE_ERROR( cudaFree( imCUDAinput ) );
	HANDLE_ERROR( cudaFree( imCUDAoutput ) );

	return 0;
}

//declare all the possible types so template compiles properly
template int medianFilterCUDASliceBySlice<unsigned char>(unsigned char* im,int* imDim,int radius,int devCUDA);
template int medianFilterCUDASliceBySlice<unsigned short int>(unsigned short int* im,int* imDim,int radius,int devCUDA);
template int medianFilterCUDASliceBySlice<float>(float* im,int* imDim,int radius,int devCUDA);