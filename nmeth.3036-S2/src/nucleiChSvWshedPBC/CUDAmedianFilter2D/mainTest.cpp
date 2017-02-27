#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include "external/Nathan/tictoc.h"
#include "medianFilter2D.h"


typedef unsigned short imageType;//define here any type you want

using namespace std;


int writeImage(void* im,long long int imSizeBytes,const char* filename)
{
	FILE* fid = fopen(filename,"wb");
	if(fid == NULL)
	{
		cout<<"ERROR: at writeImage opening file "<<filename<<endl;
		return 1;
	}
	cout<<"Writing file "<<filename<<endl;
	fwrite(im,sizeof(char),imSizeBytes,fid);
	fclose(fid);

	return 0;
}

int main( int argc, const char** argv )
{

	int imSize[dimsImageSlice+1];
	int radiusMedianFilter;
	
	imageType* im = NULL;

	int devCUDA = 0;

	//printf("Device CUDA used is %s\n",getNameDeviceCUDA(devCUDA));
	
	//setup parameters
	if(argc == 1)
	{
		imSize[0] = 410; imSize[1] = 350; 
		radiusMedianFilter = 2;
	}else if(argc == 4)
	{
		imSize[0] = atoi(argv[1]);
		imSize[1] = atoi(argv[2]);
		radiusMedianFilter = atoi(argv[3]);
	}else{
		cout<<"ERROR: at mainTest. Number of input arguments is incorrect"<<endl;
		return 2;
	}

	
	cout<<"Testing CUDA 2D median filter with radius "<<radiusMedianFilter<<" and image size "<<imSize[0]<<"x"<<imSize[1]<<endl;
	
	long long int imN = 1;	
	for(int ii=0;ii<dimsImageSlice;ii++)
	{
		imN *= (long long int) (imSize[ii]);
	}

	//allocate memory
	im = new imageType[imN];

	//fill in with random values
	/* initialize random seed: */
	srand ( time(NULL) );
	
	for(long long int ii=0;ii<imN;ii++)
		im[ii] = 100.0f*(float)(rand()/((float)RAND_MAX));

	//write out input image
	writeImage(im,imN*sizeof(imageType),"E:/temp/testMedianFilter_input.bin");
	
	//calculate convolution
	cout<<"Calculating median filter..."<<endl;
	TicTocTimer timerF=tic();
	int numIter = 1;
	for(int ii=0;ii<numIter;ii++)
	{
		if ( medianFilterCUDA(im,imSize,radiusMedianFilter,devCUDA) > 0 )
			exit(3);
	}
	cout<<"\nMedian filter calculated successfully in "<<toc(&timerF)/(float)numIter<<" secs"<<endl;
	

	//write out results
	writeImage(im,imN*sizeof(imageType),"E:/temp/testMedianFilter_result.bin");
	
	delete[] im;
	im = NULL;
	cout<<endl<<endl;
	//----------------------------------------------------------------------------------------

	//test for a whole stack
	imSize[dimsImageSlice ] = 51;
	cout<<"Testing CUDA 2D median filter with radius "<<radiusMedianFilter<<" and stack slice by slice with  size "<<imSize[0]<<"x"<<imSize[1]<<"x"<<imSize[2]<<endl;
	
	imN = 1;	
	for(int ii=0;ii<dimsImageSlice+1;ii++)
	{
		imN *= (long long int) (imSize[ii]);
	}

	//allocate memory
	im = new imageType[imN];

	//fill in with random values
	/* initialize random seed: */
	srand ( time(NULL) );
	
	for(long long int ii=0;ii<imN;ii++)
		im[ii] = 100.0f*(float)(rand()/((float)RAND_MAX));

	//write out input image
	writeImage(im,imN*sizeof(imageType),"E:/temp/testMedianFilterSliceBySlice_input.bin");
	
	//calculate convolution
	cout<<"Calculating median filter..."<<endl;
	timerF=tic();
	numIter = 1;
	for(int ii=0;ii<numIter;ii++)
	{
		if ( medianFilterCUDASliceBySlice(im,imSize,radiusMedianFilter,devCUDA) > 0 )
			exit(3);
	}
	cout<<"\nMedian filter calculated successfully in "<<toc(&timerF)/(float)numIter<<" secs"<<endl;
	

	//write out results
	writeImage(im,imN*sizeof(imageType),"E:/temp/testMedianFilterSliceBySlice_result.bin");


	//deallocate memory
	delete[] im;

	return 0;
}
