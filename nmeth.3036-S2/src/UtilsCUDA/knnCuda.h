/*
 * knnCuda.h
 *
 */

#ifndef KNNCUDA_H_
#define KNNCUDA_H_

#include "../constants.h"

int mainTestKnnCuda(void);
int allocateGPUMemoryForKnnCUDA(float *queryTemp, float **queryCUDA,
                                int **indCUDA, long long int query_nb,
                                float *scale);
void knnCUDA(int *ind, int *indCUDA, float *queryCUDA, float *refTemp,
             long long int query_nb, int ref_nb);
void knnCUDAinPlace(int *indCUDA, float *queryCUDA, float *refTemp,
                    long long int query_nb, int ref_nb);
void deallocateGPUMemoryForKnnCUDA(float **queryCUDA, int **indCUDA);
void setDeviceCUDA(int devCUDA);
void uploadScaleCUDA(float *scale);

int selectCUDAdeviceMaxMemory();

#endif /* KNNCUDA_H_ */
