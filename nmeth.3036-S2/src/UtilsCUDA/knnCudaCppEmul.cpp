/*
 * knnCudaCppEmul.cpp
 *
 *  Created on: Jul 19, 2011
 *      Author: amatf
 */
/*
 * knnCuda.cu
 *
 *  Created on: Jul 15, 2011
 *      Author: amatf
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <queue>
#include "../constants.h"

static const int MAX_REF_POINTS=3000;//we need to predefined this in order to store reference points as constant memory. Total memory needed is MAX_QUERY_POINTS*3*4 bytes. It can not be more than 5400!!!

//__constant__ float refCUDA[MAX_REF_POINTS*3];
//texture<float, 2> queryTexture2D;


struct dimIdStruct
{
	int x,y,z;
};

void knnKernelCppEmule(int *indCUDA,float *queryCUDA,int ref_nb,int query_nb,dimIdStruct threadIdx,dimIdStruct blockIdx,dimIdStruct blockDim,dimIdStruct gridDim,float *refCUDA)
{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int kMinusOne=maxGaussiansPerVoxel-1;
	int offset = blockDim.x * gridDim.x;
	float minDist[maxGaussiansPerVoxel];//to mantain distance for each index: since K is very small instead of a priority queue we keep a sorted array
	int indAux[maxGaussiansPerVoxel];
	float queryAux[dimsImage];//TODO: I can probably hardcode dimsImage to improve performance (unroll loops)

	float dist;
	int jj2;

	while(tid<query_nb)
	{
		/*texture mmemory
		queryAux[0]=tex2D(queryTexture2D,tid,0);//stores query point to compare against all the references
		queryAux[1]=tex2D(queryTexture2D,tid,1);
		queryAux[2]=tex2D(queryTexture2D,tid,2);
		*/
		queryAux[0]=queryCUDA[3*tid];
		queryAux[1]=queryCUDA[3*tid+1];
		queryAux[2]=queryCUDA[3*tid+2];

		int refIdx=0;
		for(int jj=0;jj<maxGaussiansPerVoxel;jj++) minDist[jj]=1e32;//equivalent to infinity
		for(int ii=0;ii<ref_nb;ii++)
		{
			/*
			dist=0;
			for(int jj=0;jj<dimsImage;jj++)
			{
				dist+=(queryAux[jj]-refCUDA[refIdx])*(queryAux[jj]-refCUDA[refIdx]);
				refIdx++;
			}
			*/

			dist=(queryAux[0]-refCUDA[refIdx])*(queryAux[0]-refCUDA[refIdx]);
			dist+=(queryAux[1]-refCUDA[refIdx+1])*(queryAux[1]-refCUDA[refIdx+1]);
			dist+=(queryAux[2]-refCUDA[refIdx+2])*(queryAux[2]-refCUDA[refIdx+2]);
			refIdx+=3;


			//decide weather to insert this index or not
			if(dist>minDist[kMinusOne]) continue;
			for(jj2=kMinusOne-1;jj2>=0;jj2--)
			{
				if(dist>=minDist[jj2])
				{
					minDist[jj2+1]=dist;
					indAux[jj2+1]=ii;
					break;
				}
				minDist[jj2+1]=minDist[jj2];
				indAux[jj2+1]=indAux[jj2];
			}
			if(jj2==-1)//we need to insert the element at position zero
			{
				minDist[0]=dist;
				indAux[0]=ii;
			}
		}
		//copy indexes to global memory
		jj2=tid*maxGaussiansPerVoxel;
		for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
		{
			indCUDA[jj+jj2]=indAux[jj];
		}
		//update pointer for next query_point to check
		tid+=offset;
	}
}



int mainTestKnnCudaCppEmule(void)
{

	// Variables and parameters
	float* ref;                 // Pointer to reference point array
	float* query;               // Pointer to query point array: order is x1,y1,z1,x2,y2,z2... to be cache friendly
	int*   ind;                 // Pointer to index array: size query_nb*maxGaussiansPerVoxel
	int    ref_nb     = 100;   // Reference point number
	int    query_nb   = 10;   // Query point number
	//Defined as constants now int    dimsImage        = 3;     // Dimension of points
	//int    maxGaussiansPerVoxel          = 5;     // Nearest neighbors to consider
	int    iterations = 1;     //at each iteration we will upload the query points (to simulate our case of maxGaussiansPerVoxel-NN
	int    i;

	if(MAX_REF_POINTS<ref_nb)
	{
		//TODO allow th epossibility of more ref_points by using global memory instead of constant memory
		printf("ERROR!! Increase MAX_REF_POINTS!\n");
		exit(2);
	}
	if(dimsImage!=3)
	{
		printf("ERROR: dimsImage should be 3\n");
		exit(2);
	}

	// Memory allocation
	ref    = (float *) malloc(ref_nb   * dimsImage * sizeof(float));
	query  = (float *) malloc(query_nb * dimsImage * sizeof(float));
	ind    = (int *)   malloc(query_nb * maxGaussiansPerVoxel * sizeof(float));

	// Init
	srand(time(NULL));
	for (i=0 ; i<ref_nb   * dimsImage ; i++) ref[i]    = (float)rand() / (float)RAND_MAX;
	for (i=0 ; i<query_nb * dimsImage ; i++) query[i]  = (float)rand() / (float)RAND_MAX;



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
	//----------------------------------------------------------------




	// generate a bitmap from our sphere data
	int numThreads=16;
	int numGrids=std::min(256/numThreads,(query_nb+numThreads-1)/numThreads);//TODO: play with these numbers to optimize

	//dim3    grids(numGrids,1);
	//dim3    threads(numThreads,1);
	dimIdStruct threadIdx,blockIdx,blockDim,gridDim;
	threadIdx.y=1;blockIdx.y=1;blockDim.y=1;gridDim.y=1;
	threadIdx.z=1;blockIdx.z=1;blockDim.z=1;gridDim.z=1;
	gridDim.x=numGrids;
	blockDim.x=numThreads;


	for(int ii=0;ii<iterations;ii++)
	{
		//knnKernel<<<grids,threads>>>(indCUDA,queryCUDA,ref_nb,query_nb);
		for(int jj=0;jj<numGrids;jj++)
		{
			for(int kk=0;kk<numThreads;kk++)
			{
				threadIdx.x=kk;
				blockIdx.x=jj;
				knnKernelCppEmule(ind,query,ref_nb,query_nb,threadIdx,blockIdx,blockDim,gridDim,ref);
			}
		}

		//test results
		//testKnnResults(ref,query,ind,ref_nb,query_nb,dimsImage,maxGaussiansPerVoxel);
		//update ref points (nw blob locations fater EM iterations
		for (i=0 ; i<ref_nb   * dimsImage ; i++) ref[i]    = (float)rand() / (float)RAND_MAX;
	}


	// Display informations
	printf("Number of reference points      : %6d\n", ref_nb  );
	printf("Number of query points          : %6d\n", query_nb);
	printf("Dimension of points             : %4d\n", dimsImage     );
	printf("Number of neighbors to consider : %4d\n", maxGaussiansPerVoxel       );
	printf("Processing kNN search           :\n"                );

	//----------------------------------------------------------------

	int ss=0;
	for(int kk=0;kk<query_nb;kk++)
		{
			printf("==================Checking results for query point %d==============\n",kk);
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
			{
				printf("Selected neigh CUDA id=%d\n",ind[ss]);
				ss++;
			}
		}
//-----------------------------------------------------

	// Destroy cuda event object and free memory
	free(ind);
	free(query);
	free(ref);
	return 0;
}
