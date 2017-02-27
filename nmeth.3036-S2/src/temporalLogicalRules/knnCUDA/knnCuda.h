/*
 * knnCuda.h
 *
 *  Created on: Jul 15, 2011
 *      Author: amatf
 */

#ifndef KNNCUDA_H_
#define KNNCUDA_H_


static const int MAX_REF_POINTS=5000;//we need to predefined this in order to store reference points as constant memory. Total memory needed is MAX_QUERY_POINTS*3*4 bytes. It can not be more than 5400!!!

#ifndef CUDA_MAX_SIZE_CONST //to protect agains teh same constant define in other places in the code
#define CUDA_MAX_SIZE_CONST
static const int MAX_THREADS=1024;//For Quadro4800->256;//to make sure we don't run out of registers; For TeslaC2070 -> 1024
static const int MAX_THREADS_CUDA=1024;//For Quadro4800->512;//certain kernels benefit from maximum number of threads
static const int MAX_BLOCKS=65535;
#endif

static const int maxKNN=11; //maximum number of nearest neighbors considered for the spatio-temporal graph used in tracking

#ifndef DIMS_IMAGE_CONST //to protect agains teh same constant define in other places in the code
#define DIMS_IMAGE_CONST
static const int dimsImage = 3;//to be able to precompile code
#endif


int allocateGPUMemoryForKnnCUDA_(float *queryTemp,float **queryCUDA,int **indCUDA,long long int query_nb,float *scale, int KNN);
void deallocateGPUMemoryForKnnCUDA_(float **queryCUDA,int **indCUDA);
void setDeviceCUDA_(int devCUDA);
void uploadScaleCUDA_(float *scale);

int selectCUDAdeviceMaxMemory_();


/*
\brief: simple kNN calculation. It handles all the GPU memory internally. If you need to perform kNN on the same data over and over this method is not recommeneded, but for single kNN searches it is easy to use.

\param[in] query : query points organized as x_1, x_2, x_3, ...,x_N,y_1, y_2,.....z_N  for GPU cache friendliness. Query are the points that we want to find the NN. So size(dist) = KmaxKNN * query_nb

*/
int knnCUDA_(int *ind,float* dist, float *query,float *ref,long long int query_nb,int ref_nb, int KNN, float* scale, int devCUDA);

#endif /* KNNCUDA_H_ */
