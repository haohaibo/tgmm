/*
 * knnCuda.cu
 *
 *  Created on: Jul 15, 2011
 *      Author: amatf
 */

#include <algorithm>
#include <iostream>
#include "book.h"
#include "knnCuda.h"

#if defined(_WIN32) || defined(_WIN64)
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

__constant__ float scaleCUDA[dimsImage];

int selectCUDAdeviceMaxMemory_() {
  cudaDeviceProp prop;

  int count;
  int device = -1;
  unsigned long long int totalMemMax = 0;

  HANDLE_ERROR(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    if (prop.totalGlobalMem > totalMemMax) {
      device = i;
      totalMemMax = (unsigned long)prop.totalGlobalMem;
    }
  }

  if (device < -1)
    printf("ERROR: at selectCUDAdeviceMaxMemory(): CUDA device not selected\n");

  return device;
}

__device__ inline void findMaxPosition(float *distArray, float *minDist,
                                       int *pos, int KNN) {
  (*minDist) = distArray[0];
  (*pos) = 0;
  for (int ii = 1; ii < KNN; ii++) {
    if ((*minDist) < distArray[ii]) {
      (*minDist) = distArray[ii];
      (*pos) = ii;
    }
  }
}

//===========================================================================================
__global__ void __launch_bounds__(MAX_THREADS)
    knnKernelNoConstantMemory(int *indCUDA, float *distCUDA, float *queryCUDA,
                              float *anchorCUDA, int ref_nb,
                              long long int query_nb, int KNN) {
  // map from threadIdx/BlockIdx to pixel position
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // int offset = blockDim.x * gridDim.x;

  if (KNN > maxKNN) return;  // code is not ready for this
  if (tid >= query_nb) return;

  // int kMinusOne=maxKNN-1;
  float minDist[maxKNN];  // to mantain distance for each index: since K is very
                          // small instead of a priority queue we keep a sorted
                          // array
  int indAux[maxKNN];
  float queryAux[dimsImage];
  float minDistThr;

  float dist, distAux;
  int jj2, minPos;

  jj2 = tid;

  // global memory: organized as x_1,x_2,x_3,....,y_1,y_2,...,z_1,... to have
  // coalescent access
  queryAux[0] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[1] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[2] = queryCUDA[jj2];

  int refIdx;
  for (int jj = 0; jj < KNN; jj++)
    minDist[jj] = 1e32;  // equivalent to infinity. Thus, we know this element
                         // has not been assigned
  minDistThr = 1e32;
  minPos = 0;
  for (int ii = 0; ii < ref_nb; ii++) {
    //__syncthreads();//to access constant memory coherently (this was effective
    // in CUDA 3.2)
    refIdx = ii;

    distAux = (queryAux[0] - anchorCUDA[refIdx]) * scaleCUDA[0];
    dist = distAux * distAux;
    refIdx += ref_nb;
    if (dist > minDistThr) continue;
    distAux = (queryAux[1] - anchorCUDA[refIdx]) * scaleCUDA[1];
    dist += distAux * distAux;
    refIdx += ref_nb;
    if (dist > minDistThr) continue;
    distAux = (queryAux[2] - anchorCUDA[refIdx]) * scaleCUDA[2];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;

    // insert element" minimize memory exchanges
    minDist[minPos] = dist;
    indAux[minPos] = ii;
    findMaxPosition(minDist, &minDistThr, &minPos, KNN);
  }

  __syncthreads();  // I need this to have coalescent memory access to inCUDA:
                    // speeds up the code by x4

  // copy indexes to global memory
  jj2 = tid;
  for (int jj = 0; jj < KNN; jj++) {
    // indCUDA[jj+jj2]=indAux[jj];
    indCUDA[jj2] = indAux[jj];
    jj2 += query_nb;
  }
  // copy distance if requested by user
  if (distCUDA != NULL) {
    jj2 = tid;
    for (int jj = 0; jj < KNN; jj++) {
      // indCUDA[jj+jj2]=indAux[jj];
      distCUDA[jj2] = minDist[jj];
      jj2 += query_nb;
    }
  }

  // update pointer for next query_point to check
  // tid+=offset;
}

//=============================================================================================================
int knnCUDA_(int *ind, float *dist, float *query, float *ref,
             long long int query_nb, int ref_nb, int KNN, float *scale,
             int devCUDA) {
  // Variables and parameters
  // float* ref;                 // Pointer to reference point array: order is
  // cache friednly with the GPU
  // float* query;               // Pointer to query point array: order is
  // x1,y1,z1,x2,y2,z2... to be cache friendly
  // int*   ind;                 // Pointer to index array: size
  // query_nb*maxKNN. Again, order is GPU cache friendly.
  // float*   dist;              // Pointer to distance^2 array: size
  // query_nb*maxKNN. Again, order is GPU cache friendly. If pointer is null,
  // scaled euclidean distance to each nearest neighbor is not returned
  // int    ref_nb       // Reference point number
  // int    query_nb    // Query point number
  // float scale[dimsImage] //

  if (dimsImage != 3) {
    printf(
        "ERROR: at knnCUDA: code is not ready for dimsImage other than 3\n");  // TODO: change this to any dimensionality
    return 2;
  }

  if (ref_nb <= 0)  // nothing to do. There are no possible assignments
  {
    if (dist != NULL) {
      for (long long int ii = 0; ii < query_nb * KNN; ii++)
        dist[ii] = 1e32f;  // no assignments
    }
    return 0;
  }
  // CUDA variables
  int *indCUDA;
  float *queryCUDA;
  float *anchorCUDA;
  float *distCUDA = NULL;

  // set CUDA device
  HANDLE_ERROR(cudaSetDevice(devCUDA));

  // allocate memory on the GPU for the output: it will only be done once in the
  // whole program
  HANDLE_ERROR(cudaMalloc(
      (void **)&indCUDA,
      query_nb * KNN * sizeof(int)));  // should it be a texture memory?NO. It
                                       // does not fit in Cuda2Darray but it
                                       // fits in linear 1Dtexture, although it
                                       // does not seems to bring benefits
  HANDLE_ERROR(
      cudaMalloc((void **)&queryCUDA, query_nb * dimsImage * sizeof(float)));
  HANDLE_ERROR(
      cudaMalloc((void **)&anchorCUDA, ref_nb * dimsImage * sizeof(float)));
  if (dist != NULL)
    HANDLE_ERROR(
        cudaMalloc((void **)&distCUDA, query_nb * KNN * sizeof(float)));

  // Copy image data to array
  HANDLE_ERROR(cudaMemcpy(queryCUDA, query,
                          dimsImage * query_nb * sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(anchorCUDA, ref, dimsImage * ref_nb * sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(
      scaleCUDA, scale, dimsImage * sizeof(float)));  // constant memory

  // prepare to launch kernel
  int numThreads = min(MAX_THREADS, (int)query_nb);
  int numGrids = min(
      MAX_BLOCKS, (int)(query_nb + numThreads - 1) /
                      numThreads);  // TODO: play with these numbers to optimize

  knnKernelNoConstantMemory<<<numGrids, numThreads>>>(
      indCUDA, distCUDA, queryCUDA, anchorCUDA, ref_nb, query_nb, KNN);
  HANDLE_ERROR_KERNEL;

  // copy results back
  HANDLE_ERROR(cudaMemcpy(ind, indCUDA, query_nb * KNN * sizeof(int),
                          cudaMemcpyDeviceToHost));  // retrieve indexes:
                                                     // memcopy is synchronous
                                                     // unless stated otherwise
  if (distCUDA != NULL)
    HANDLE_ERROR(cudaMemcpy(dist, distCUDA, query_nb * KNN * sizeof(float),
                            cudaMemcpyDeviceToHost));

  // free memory
  HANDLE_ERROR(cudaFree(indCUDA));
  HANDLE_ERROR(cudaFree(queryCUDA));
  HANDLE_ERROR(cudaFree(anchorCUDA));
  if (distCUDA != NULL) HANDLE_ERROR(cudaFree(distCUDA));
  return 0;
}

//===================================================================================================
int allocateGPUMemoryForKnnCUDA_(float *queryTemp, float **queryCUDA,
                                 int **indCUDA, long long int query_nb,
                                 float *scale, int KNN) {
  HANDLE_ERROR(cudaMalloc((void **)&(*indCUDA), query_nb * KNN * sizeof(int)));
  HANDLE_ERROR(
      cudaMalloc((void **)&(*queryCUDA), query_nb * dimsImage * sizeof(float)));
  // Copy image data to array
  HANDLE_ERROR(cudaMemcpy((*queryCUDA), queryTemp,
                          dimsImage * query_nb * sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(
      scaleCUDA, scale, dimsImage * sizeof(float)));  // constant memory
  return 0;
}

void setDeviceCUDA_(int devCUDA) {
  // WE ASSUME qeuryCUDA AND indCUDA HAVE BEEN ALLOCATED ALREADY AND MEMORY
  // TRANSFERRED TO THE GPU
  HANDLE_ERROR(cudaSetDevice(devCUDA));
}

//====================================================================================================
void deallocateGPUMemoryForKnnCUDA_(float **queryCUDA, int **indCUDA) {
  HANDLE_ERROR(cudaFree(*indCUDA));
  (*indCUDA) = NULL;
  HANDLE_ERROR(cudaFree(*queryCUDA));
  (*queryCUDA) = NULL;
}
//==============================================================
void uploadScaleCUDA_(float *scale) {
  HANDLE_ERROR(cudaMemcpyToSymbol(
      scaleCUDA, scale, dimsImage * sizeof(float)));  // constant memory
}
