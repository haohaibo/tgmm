/*
 * knnCuda.cu
 *
 */

//#define USE_CUDA_PRINTF
#include <algorithm>
#include <iostream>
#include "GMEMcommonCUDA.h"
#include "external/book.h"
#include "knnCuda.h"
#include "testKnnResults.h"  //to debug results (remove after we know the code works)

#if defined(_WIN32) || defined(_WIN64)
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#ifdef USE_CUDA_PRINTF
#include "external/cuPrintf.cu"
#endif

__constant__ float refCUDA[MAX_REF_POINTS * 3];
__constant__ float scaleCUDA[dimsImage];

int selectCUDAdeviceMaxMemory() {
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
                                       int *pos) {
  (*minDist) = distArray[0];
  (*pos) = 0;
  for (int ii = 1; ii < maxGaussiansPerVoxel; ii++) {
    if ((*minDist) < distArray[ii]) {
      (*minDist) = distArray[ii];
      (*pos) = ii;
    }
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    knnKernel(int *indCUDA, float *queryCUDA, int ref_nb,
              long long int query_nb) {
  // map from threadIdx/BlockIdx to pixel position
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // int offset = blockDim.x * gridDim.x;

  if (tid >= query_nb) return;

  // int kMinusOne=maxGaussiansPerVoxel-1;
  float minDist[maxGaussiansPerVoxel];  // to mantain distance for each index:
                                        // since K is very small instead of a
                                        // priority queue we keep a sorted array
  int indAux[maxGaussiansPerVoxel];
  float queryAux[dimsImage];
  float minDistThr;
  // float scaleAux[dimsImage];
  // scaleAux[0]=scaleCUDA[0];scaleAux[1]=scaleCUDA[1];scaleAux[2]=scaleCUDA[2];

  float dist, distAux;
  int jj2, minPos;

  jj2 = tid;
  /*texture mmemory
          queryAux[0]=tex1Dfetch(queryTexture,jj2);//stores query point to
     compare against all the references
          queryAux[1]=tex1Dfetch(queryTexture,jj2+1);
          queryAux[2]=tex1Dfetch(queryTexture,jj2+2);
   */

  // global memory: organized as x_1,x_2,x_3,....,y_1,y_2,...,z_1,... to have
  // coalescent access
  queryAux[0] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[1] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[2] = queryCUDA[jj2];

  int refIdx = -3;
  for (int jj = 0; jj < maxGaussiansPerVoxel; jj++)
    minDist[jj] = 1e32;  // equivalent to infinity
  minDistThr = 1e32;
  minPos = 0;
  for (int ii = 0; ii < ref_nb; ii++) {
    __syncthreads();  // to access constant memory coherently
    refIdx += 3;
    /*
            dist=0;
            for(int jj=0;jj<dimsImage;jj++)
            {
                    dist+=(queryAux[jj]-refCUDA[refIdx])*(queryAux[jj]-refCUDA[refIdx]);
                    refIdx++;
            }
     */

    distAux = (queryAux[0] - refCUDA[refIdx]) * scaleCUDA[0];
    dist = distAux * distAux;
    if (dist > minDistThr) continue;
    distAux = (queryAux[1] - refCUDA[refIdx + 1]) * scaleCUDA[1];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;
    distAux = (queryAux[2] - refCUDA[refIdx + 2]) * scaleCUDA[2];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;

    // insert element" minimize memory exchanges
    minDist[minPos] = dist;
    indAux[minPos] = ii;
    findMaxPosition(minDist, &minDistThr, &minPos);
  }

  __syncthreads();  // I need this to have coalescent memory access to inCUDA:
                    // speeds up the code by x4

  // copy indexes to global memory
  jj2 = tid;
  for (int jj = 0; jj < maxGaussiansPerVoxel; jj++) {
    // indCUDA[jj+jj2]=indAux[jj];
    indCUDA[jj2] = indAux[jj];
    jj2 += query_nb;
  }
  // update pointer for next query_point to check
  // tid+=offset;
}

//===========================================================================================
__global__ void __launch_bounds__(MAX_THREADS)
    knnKernelNoConstantMemory(int *indCUDA, float *queryCUDA, float *anchorCUDA,
                              int ref_nb, long long int query_nb) {
  // map from threadIdx/BlockIdx to pixel position
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // int offset = blockDim.x * gridDim.x;

  if (tid >= query_nb) return;

  // int kMinusOne=maxGaussiansPerVoxel-1;
  float minDist[maxGaussiansPerVoxel];  // to mantain distance for each index:
                                        // since K is very small instead of a
                                        // priority queue we keep a sorted array
  int indAux[maxGaussiansPerVoxel];
  float queryAux[dimsImage];
  float minDistThr;
  // float scaleAux[dimsImage];
  // scaleAux[0]=scaleCUDA[0];scaleAux[1]=scaleCUDA[1];scaleAux[2]=scaleCUDA[2];

  float dist, distAux;
  int jj2, minPos;

#ifdef USE_CUDA_PRINTF
  if (tid == -1) cuPrintf("tid=%d\n", tid);
#endif

  jj2 = tid;
  /*texture mmemory
          queryAux[0]=tex1Dfetch(queryTexture,jj2);//stores query point to
     compare against all the references
          queryAux[1]=tex1Dfetch(queryTexture,jj2+1);
          queryAux[2]=tex1Dfetch(queryTexture,jj2+2);
   */

  // global memory: organized as x_1,x_2,x_3,....,y_1,y_2,...,z_1,... to have
  // coalescent access
  queryAux[0] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[1] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[2] = queryCUDA[jj2];

#ifdef USE_CUDA_PRINTF
  if (tid == -1) {
    cuPrintf("tid=%d;queryAux=%f %f %f\n", tid, queryAux[0], queryAux[1],
             queryAux[2]);
  }
#endif
  int refIdx = -3;
  for (int jj = 0; jj < maxGaussiansPerVoxel; jj++)
    minDist[jj] = 1e32;  // equivalent to infinity
  minDistThr = 1e32;
  minPos = 0;
  for (int ii = 0; ii < ref_nb; ii++) {
    __syncthreads();  // to access constant memory coherently
    refIdx += 3;
    /*
            dist=0;
            for(int jj=0;jj<dimsImage;jj++)
            {
                    dist+=(queryAux[jj]-anchorCUDA[refIdx])*(queryAux[jj]-anchorCUDA[refIdx]);
                    refIdx++;
            }
     */

    distAux = (queryAux[0] - anchorCUDA[refIdx]) * scaleCUDA[0];
    dist = distAux * distAux;
    if (dist > minDistThr) continue;
    distAux = (queryAux[1] - anchorCUDA[refIdx + 1]) * scaleCUDA[1];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;
    distAux = (queryAux[2] - anchorCUDA[refIdx + 2]) * scaleCUDA[2];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;
#ifdef USE_CUDA_PRINTF
    if (tid == -1) {
      cuPrintf("tid=%d;refIdx-3=%d;anchorCUDA=%f %f %f\n", tid, refIdx - 3,
               anchorCUDA[refIdx - 3], anchorCUDA[refIdx - 2],
               anchorCUDA[refIdx - 1]);
    }
#endif
    // insert element" minimize memory exchanges
    minDist[minPos] = dist;
    indAux[minPos] = ii;
    findMaxPosition(minDist, &minDistThr, &minPos);
  }

  __syncthreads();  // I need this to have coalescent memory access to inCUDA:
                    // speeds up the code by x4

  // copy indexes to global memory
  jj2 = tid;
  for (int jj = 0; jj < maxGaussiansPerVoxel; jj++) {
    // indCUDA[jj+jj2]=indAux[jj];
    indCUDA[jj2] = indAux[jj];
    jj2 += query_nb;
  }
  // update pointer for next query_point to check
  // tid+=offset;
}
//===========================================================================================
__device__ inline void Comparator(float &keyA, int &valA, float &keyB,
                                  int &valB, unsigned int dir) {
  float t;
  int v;
  if ((keyA > keyB) == dir) {
    t = keyA;
    keyA = keyB;
    keyB = t;
    v = valA;
    valA = valB;
    valB = v;
  }
}
// this kernel needs to be called with MAX_THREADS_CUDA
__global__ void __launch_bounds__(MAX_THREADS_CUDA)
    knnKernelSorting(int *indCUDA, float *queryCUDA, float *anchorCUDA,
                     int ref_nb, long long int query_nb) {
  // Shared memory storage for one or more short vectors
  __shared__ float s_key[MAX_THREADS_CUDA];  // distance
  __shared__ int s_val[MAX_THREADS_CUDA];    // index
  __shared__ int indAux[maxGaussiansPerVoxel];
  __shared__ float minDist[maxGaussiansPerVoxel];

  float x_n[dimsImage];
  unsigned int dir = 0;  // ascending order sorting

  // map from threadIdx/BlockIdx to pixel position
  long long int tid = blockIdx.x;
  long long int pos2;
  float dist, aux;

  int maxOffset =
      ((ref_nb + MAX_THREADS_CUDA - 1) / MAX_THREADS_CUDA) * MAX_THREADS_CUDA;

  while (tid < query_nb) {
    // load query point
    pos2 = tid;
    x_n[0] = queryCUDA[pos2];
    pos2 += query_nb;
    x_n[1] = queryCUDA[pos2];
    pos2 += query_nb;
    x_n[2] = queryCUDA[pos2];

    for (int offset = threadIdx.x; offset < maxOffset;
         offset += MAX_THREADS_CUDA) {
      // calculate distance
      if (offset < ref_nb) {
        aux = x_n[0] - anchorCUDA[offset];
        dist = aux * aux;
        offset += ref_nb;
        aux = x_n[0] - anchorCUDA[offset];
        dist += aux * aux;
        offset += ref_nb;
        aux = x_n[2] - anchorCUDA[offset];
        dist += aux * aux;
      } else {
        dist = 1e32;
      }
      s_val[threadIdx.x] = offset;
      s_key[threadIdx.x] = dist;
      __syncthreads();

      // sort value and key in shared memory: bitonc search from Cuda SDK
      for (unsigned int size = 2; size < MAX_THREADS_CUDA; size <<= 1) {
        // Bitonic merge
        unsigned int ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);
        for (unsigned int stride = size / 2; stride > 0; stride >>= 1) {
          __syncthreads();
          unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
          Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                     s_val[pos + stride], ddd);
        }
      }

      // ddd == dir for the last bitonic merge step
      {
        for (unsigned int stride = MAX_THREADS_CUDA / 2; stride > 0;
             stride >>= 1) {
          __syncthreads();
          unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
          Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                     s_val[pos + stride], dir);
        }
      }
      __syncthreads();

      // merge this batch of distances with short sorted array
      if (offset <
          maxGaussiansPerVoxel)  // we just need to copy in the first iteration
      {
        indAux[offset] = s_val[offset];
        minDist[offset] = s_key[offset];
      } else {
        if (threadIdx.x == 0) {  // merge two sorted arrays
          int ptr1 = 0;
          // int	ptr2=0;
          while (ptr1 < maxGaussiansPerVoxel) {
            ptr1++;
            // TODO: finish this although the kernel is really slow (in
            // comparison) even without this part
          }
        }
      }
      __syncthreads();
    }
    // copy indexes to global memory
    if (threadIdx.x < maxGaussiansPerVoxel) {
      indCUDA[tid + threadIdx.x * query_nb] =
          indAux[threadIdx.x];  // not coalescence
    }
    // update pointer for next query_point to check
    tid += gridDim.x;
    __syncthreads();
  }
}

__global__ void __launch_bounds__(MAX_THREADS)
    knnKernelSortedArray(int *indCUDA, float *queryCUDA, int ref_nb,
                         long long int query_nb) {
  // map from threadIdx/BlockIdx to pixel position
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // int offset = blockDim.x * gridDim.x;

  if (tid >= query_nb) return;

  int kMinusOne = maxGaussiansPerVoxel - 1;
  float minDist[maxGaussiansPerVoxel];  // to mantain distance for each index:
                                        // since K is very small instead of a
                                        // priority queue we keep a sorted array
  int indAux[maxGaussiansPerVoxel];
  float queryAux[dimsImage];  // TODO: I can probably hardcode dimsImage to
                              // improve performance (unroll loops)
  float minDistThr;
  float scaleAux[dimsImage];
  scaleAux[0] = scaleCUDA[0];
  scaleAux[1] = scaleCUDA[1];
  scaleAux[2] = scaleCUDA[2];

  float dist, distAux;
  int jj2;

#ifdef USE_CUDA_PRINTF
  if (tid == 0) cuPrintf("tid=%d\n", tid);
#endif

  jj2 = tid;
  /*texture mmemory
          queryAux[0]=tex1Dfetch(queryTexture,jj2);//stores query point to
     compare against all the references
          queryAux[1]=tex1Dfetch(queryTexture,jj2+1);
          queryAux[2]=tex1Dfetch(queryTexture,jj2+2);
   */

  // global memory: organized as x_1,x_2,x_3,....,y_1,y_2,...,z_1,... to have
  // coalescent access
  queryAux[0] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[1] = queryCUDA[jj2];
  jj2 += query_nb;
  queryAux[2] = queryCUDA[jj2];

#ifdef USE_CUDA_PRINTF
  if (tid == 1) {
    cuPrintf("tid=%d;queryAux=%f %f %f\n", tid, queryAux[0], queryAux[1],
             queryAux[2]);
  }
#endif
  int refIdx = -3;
  for (int jj = 0; jj < maxGaussiansPerVoxel; jj++)
    minDist[jj] = 1e32;  // equivalent to infinity
  minDistThr = minDist[kMinusOne];
  for (int ii = 0; ii < ref_nb; ii++) {
    __syncthreads();  // to access constant memory coherently
    refIdx += 3;
    /*
            dist=0;
            for(int jj=0;jj<dimsImage;jj++)
            {
                    dist+=(queryAux[jj]-refCUDA[refIdx])*(queryAux[jj]-refCUDA[refIdx]);
                    refIdx++;
            }
     */

    distAux = (queryAux[0] - refCUDA[refIdx]) * scaleAux[0];
    dist = distAux * distAux;
    if (dist > minDistThr) continue;
    distAux = (queryAux[1] - refCUDA[refIdx + 1]) * scaleAux[1];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;
    distAux = (queryAux[2] - refCUDA[refIdx + 2]) * scaleAux[2];
    dist += distAux * distAux;
    if (dist > minDistThr) continue;
#ifdef USE_CUDA_PRINTF
    if (tid == 1) {
      cuPrintf("tid=%d;refIdx-3=%d;refCuda=%f %f %f\n", tid, refIdx - 3,
               refCUDA[refIdx - 3], refCUDA[refIdx - 2], refCUDA[refIdx - 1]);
    }
#endif
    // decide weather to insert this index or not
    for (jj2 = kMinusOne - 1; jj2 >= 0; jj2--) {
      if (dist >= minDist[jj2]) {
        minDist[jj2 + 1] = dist;
        indAux[jj2 + 1] = ii;
        break;
      }
      minDist[jj2 + 1] = minDist[jj2];
      indAux[jj2 + 1] = indAux[jj2];
    }
    if (jj2 == -1)  // we need to insert the element at position zero
    {
      minDist[0] = dist;
      indAux[0] = ii;
    }
    minDistThr = minDist[kMinusOne];
  }

  __syncthreads();  // I need this to have coalescent memory access to inCUDA:
                    // speeds up the code by x4

  // copy indexes to global memory
  jj2 = tid;
  for (int jj = 0; jj < maxGaussiansPerVoxel; jj++) {
    // indCUDA[jj+jj2]=indAux[jj];
    indCUDA[jj2] = indAux[jj];
    jj2 += query_nb;
  }
  // update pointer for next query_point to check
  // tid+=offset;
}

//========================================================================================================================================
/*

int main()
{
        return mainTestKnnCuda();
}
*/

int mainTestKnnCuda(void) {
  // Variables and parameters
  float *ref;    // Pointer to reference point array
  float *query;  // Pointer to query point array: order is x1,y1,z1,x2,y2,z2...
                 // to be cache friendly
  int *ind;      // Pointer to index array: size query_nb*maxGaussiansPerVoxel
  int ref_nb = 1377;      // Reference point number
  int query_nb = 100000;  // Query point number
  // Defined as constants now int    dimsImage        = 3;     // Dimension of
  // points
  // int    maxGaussiansPerVoxel          = 5;     // Nearest neighbors to
  // consider
  int iterations = 5;  // at each iteration we will upload the query points (to
                       // simulate our case of maxGaussiansPerVoxel-NN
  int i;

  if (MAX_REF_POINTS < ref_nb) {
    // TODO allow th epossibility of more ref_points by using global memory
    // instead of constant memory
    printf("ERROR!! Increase MAX_REF_POINTS!\n");
    exit(2);
  }
  if (dimsImage != 3) {
    printf("ERROR: dimsImage should be 3\n");
    exit(2);
  }

  // Memory allocation
  ref = (float *)malloc(ref_nb * dimsImage * sizeof(float));
  query = (float *)malloc(query_nb * dimsImage * sizeof(float));
  ind = (int *)malloc(query_nb * maxGaussiansPerVoxel * sizeof(int));

  // Init
  srand((unsigned int)time(NULL));
  for (i = 0; i < ref_nb * dimsImage; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dimsImage; i++)
    query[i] = (float)rand() / (float)RAND_MAX;

  // CUDA variables
  int *indCUDA;
  float *queryCUDA;
  float scale[dimsImage] = {1.0, 1.0, 1.0};
  float *anchorCUDA;
  // cudaArray* queryCUDA;

  // select GPU and check maximum meory available and check that we have enough
  // memory
  int memoryNeededInBytes = query_nb * maxGaussiansPerVoxel * sizeof(int) +
                            query_nb * dimsImage * sizeof(float);
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.totalGlobalMem = memoryNeededInBytes;
  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
  printf("Memory required: %d;CUDA choosing device:  %d\n", memoryNeededInBytes,
         dev);
  HANDLE_ERROR(cudaSetDevice(dev));
/*
memoryNeededInBytes=ref_nb   * dimsImage * sizeof(float);//make sure we can
allocate texture memory
if(((double)memoryNeededInBytes)/1048576.0 > MAX_TEXTURE1D_MEM)
{
        printf("ERROR: number of query points exceeds maximum texture memory.
Code is not ready for this\n");
        exit(2);
}
*/
#ifdef USE_CUDA_PRINTF
  cudaPrintfInit();
#endif

  // allocate memory on the GPU for the output: it will only be done once in the
  // whole program
  HANDLE_ERROR(cudaMalloc((void **)&indCUDA,
                          query_nb * maxGaussiansPerVoxel *
                              sizeof(int)));  // should it be a texture
                                              // memory?NO. It does not fit in
                                              // Cuda2Darray but it fits in
                                              // linear 1Dtexture, although it
  // does not seems to bring benefits
  HANDLE_ERROR(
      cudaMalloc((void **)&queryCUDA, query_nb * dimsImage * sizeof(float)));
  HANDLE_ERROR(
      cudaMalloc((void **)&anchorCUDA, ref_nb * dimsImage * sizeof(float)));

  // capture the start time
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // queryCUDA is global memory
  // HANDLE_ERROR(cudaMemcpy(queryCUDA,query, dimsImage*query_nb*sizeof(float),
  // cudaMemcpyHostToDevice));

  /*texture memory
  //queryCUDA is 1D texture memory
  cudaChannelFormatDesc description = cudaCreateChannelDesc<float>();
    // Bind the array to the texture
    HANDLE_ERROR( cudaBindTexture(NULL,queryTexture,
  queryCUDA,dimsImage*query_nb*sizeof(float)) );
    */

  // Copy image data to array
  HANDLE_ERROR(cudaMemcpy(queryCUDA, query,
                          dimsImage * query_nb * sizeof(float),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpyToSymbol(
      scaleCUDA, scale, dimsImage * sizeof(float)));  // constant memory

  // generate a bitmap from our sphere data
  int numThreads = min(MAX_THREADS, query_nb);
  int numGrids = min(
      MAX_BLOCKS, (query_nb + numThreads - 1) /
                      numThreads);  // TODO: play with these numbers to optimize
  printf("NumThreads=%d;numGrids=%d\n", numThreads, numGrids);
  dim3 grids(numGrids, 1);
  dim3 threads(numThreads, 1);
  for (int ii = 0; ii < iterations; ii++) {
    // update ref points (nw blob locations fater EM iterations
    for (i = 0; i < ref_nb * dimsImage; i++)
      ref[i] = (float)rand() / (float)RAND_MAX;

    HANDLE_ERROR(cudaMemcpyToSymbol(
        refCUDA, ref, ref_nb * dimsImage * sizeof(float)));  // constant memory
    knnKernel<<<grids, threads>>>(indCUDA, queryCUDA, ref_nb, query_nb);

    /*
    int *indHost=new int[maxGaussiansPerVoxel*query_nb];
    HANDLE_ERROR(cudaMemcpy(indHost,indCUDA,
    query_nb*maxGaussiansPerVoxel*sizeof(int), cudaMemcpyDeviceToHost));
    printf("==================================\n");
    for(int ss=0;ss<query_nb;ss++)
    {
            for(int rr=0;rr<maxGaussiansPerVoxel;rr++)
            {
                    printf("%d ",indHost[ss+query_nb*rr]);
            }
            printf(";\n");
    }
    */
    // HANDLE_ERROR(cudaMemcpy(anchorCUDA,ref, dimsImage*ref_nb*sizeof(float),
    // cudaMemcpyHostToDevice));
    // knnKernelNoConstantMemory<<<grids,threads>>>(indCUDA,queryCUDA,anchorCUDA,ref_nb,query_nb);

    // HANDLE_ERROR(cudaMemcpy(anchorCUDA,ref, dimsImage*ref_nb*sizeof(float),
    // cudaMemcpyHostToDevice));
    // knnKernelSorting<<<min(query_nb,MAX_BLOCKS),MAX_THREADS_CUDA>>>(indCUDA,queryCUDA,anchorCUDA,ref_nb,query_nb);

    HANDLE_ERROR_KERNEL;
    HANDLE_ERROR(cudaMemcpy(
        ind, indCUDA, query_nb * maxGaussiansPerVoxel * sizeof(int),
        cudaMemcpyDeviceToHost));  // retrieve indexes: memcopy is synchronous
                                   // unless stated otherwise

#ifdef USE_CUDA_PRINTF
    cudaPrintfDisplay(stdout, true);
#endif

    // test results
    // testKnnResults(ref,query,ind,ref_nb,query_nb,dimsImage,maxGaussiansPerVoxel);
  }

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dimsImage);
  printf("Number of neighbors to consider : %4d\n", maxGaussiansPerVoxel);
  printf("Processing kNN search           :\n");
  // get stop time, and display the timing results
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsedTime;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf(" done in %f secs for %d iterations (%f secs per iteration)\n",
         elapsedTime / 1000, iterations, elapsedTime / (iterations * 1000));

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));

  HANDLE_ERROR(cudaFree(indCUDA));
  // Unbind the array from the texture
  // cudaUnbindTexture(queryTexture);

  HANDLE_ERROR(cudaFree(queryCUDA));

#ifdef USE_CUDA_PRINTF
  cudaPrintfEnd();
#endif

  /*
  //---------------------------print out debug ---------------
          printf("query=[\n");
          int ss2=0;
          for(int ii=0;ii<query_nb;ii++)
          {
                  printf("%f %f %f;\n",query[ss2],query[ss2+1],query[ss2+2]);
                  ss2+=dimsImage;
          }
          printf("];\n");

                  printf("ref=[\n");
          ss2=0;
          for(int ii=0;ii<ref_nb;ii++)
          {
                  printf("%f %f %f;\n",ref[ss2],ref[ss2+1],ref[ss2+2]);
                  ss2+=dimsImage;
          }
          printf("];\n");


          int ss=0;
          for(int kk=0;kk<query_nb;kk++)
                  {
                          printf("==================Checking results for query
  point %d==============\n",kk);
                          for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
                          {
                                  printf("Selected neigh CUDA id=%d\n",ind[ss]);
                                  ss++;
                          }
                  }
  //-----------------------------------------------------
  */

  // Destroy cuda event object and free memory
  free(ind);
  free(query);
  free(ref);
  return 0;
}

//===================================================================================================
int allocateGPUMemoryForKnnCUDA(float *queryTemp, float **queryCUDA,
                                int **indCUDA, long long int query_nb,
                                float *scale) {
  HANDLE_ERROR(cudaMalloc((void **)&(*indCUDA),
                          query_nb * maxGaussiansPerVoxel * sizeof(int)));
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

void setDeviceCUDA(int devCUDA) {
  // WE ASSUME qeuryCUDA AND indCUDA HAVE BEEN ALLOCATED ALREADY AND MEMORY
  // TRANSFERRED TO THE GPU
  HANDLE_ERROR(cudaSetDevice(devCUDA));
}
//===================================================================================================
void knnCUDA(int *ind, int *indCUDA, float *queryCUDA, float *refTemp,
             long long int query_nb, int ref_nb) {
#ifdef USE_CUDA_PRINTF
  cudaPrintfInit();
#endif
  if (MAX_REF_POINTS < ref_nb) {
    // TODO allow th epossibility of more ref_points by using global memory
    // instead of constant memory
    printf("ERROR!! Increase MAX_REF_POINTS!\n");
    exit(2);
  }
  if (dimsImage != 3) {
    printf("ERROR: dimsImage should be 3\n");
    exit(2);
  }
  // calculate number of threads and blocks
  long long int numThreads = std::min((long long int)MAX_THREADS, query_nb);
  long long int numGrids = std::min((long long int)MAX_BLOCKS,
                                    (query_nb + numThreads - 1) / numThreads);

  // printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);

  HANDLE_ERROR(cudaMemcpyToSymbol(
      refCUDA, refTemp,
      ref_nb * dimsImage * sizeof(float)));  // constant memory
  knnKernel<<<numGrids, numThreads>>>(indCUDA, queryCUDA, ref_nb, query_nb);
  HANDLE_ERROR_KERNEL;
#ifdef USE_CUDA_PRINTF
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif

  HANDLE_ERROR(cudaMemcpy(
      ind, indCUDA, query_nb * maxGaussiansPerVoxel * sizeof(int),
      cudaMemcpyDeviceToHost));  // retrieve indexes: memcopy is synchronous
                                 // unless stated otherwise
}
//===================================================================================================
void knnCUDAinPlace(int *indCUDA, float *queryCUDA, float *refTemp,
                    long long int query_nb, int ref_nb) {
#ifdef USE_CUDA_PRINTF
  cudaPrintfInit();
#endif

  if (dimsImage != 3) {
    printf("ERROR: dimsImage should be 3\n");
    exit(2);
  }
  // calculate number of threads and blocks
  long long int numThreads = std::min((long long int)MAX_THREADS, query_nb);
  long long int numGrids = std::min((long long int)MAX_BLOCKS,
                                    (query_nb + numThreads - 1) / numThreads);

  // printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);

  if (MAX_REF_POINTS < ref_nb)  // use global memory for anchorPoints (slower)
  {
    knnKernelNoConstantMemory<<<numGrids, numThreads>>>(
        indCUDA, queryCUDA, refTemp, ref_nb, query_nb);
    HANDLE_ERROR_KERNEL;
  } else {  // use constant memory for anchor points
    HANDLE_ERROR(
        cudaMemcpyToSymbol(refCUDA, refTemp, ref_nb * dimsImage * sizeof(float),
                           0, cudaMemcpyDeviceToDevice));  // constant memory
    knnKernel<<<numGrids, numThreads>>>(indCUDA, queryCUDA, ref_nb, query_nb);
    HANDLE_ERROR_KERNEL;
  }
#ifdef USE_CUDA_PRINTF
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
#endif
  // it is inplace so we don't copy back to host
  // HANDLE_ERROR(cudaMemcpy(ind,indCUDA,query_nb*maxGaussiansPerVoxel*sizeof(int),cudaMemcpyDeviceToHost));//retrieve
  // indexes: memcopy is synchronous unless stated otherwise
}

//====================================================================================================
void deallocateGPUMemoryForKnnCUDA(float **queryCUDA, int **indCUDA) {
  HANDLE_ERROR(cudaFree(*indCUDA));
  (*indCUDA) = NULL;
  HANDLE_ERROR(cudaFree(*queryCUDA));
  (*queryCUDA) = NULL;
}
//==============================================================
void uploadScaleCUDA(float *scale) {
  HANDLE_ERROR(cudaMemcpyToSymbol(
      scaleCUDA, scale, dimsImage * sizeof(float)));  // constant memory
}
