
/*
* Copyright (C) 2011-2012 by  Fernando Amat
* See license.txt for full license and copyright notice.
*
* Authors: Fernando Amat 
*  watershedPersistanceAgglomeration.cpp
*
*  Created on: September 17th, 2012
*      Author: Fernando Amat
*
* \brief Implements main functionalities to segment 
* nuclei channel using watershed + agglomeration based 
* on persistance methods
*
*/
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <stdio.h>
#include <stdlib.h>
//hash_tabel implementation 
//(it should be faster than maps (they are trees)
#include <unordered_map> 

//hash_tabel implementation 
//(it should be faster than maps (they are trees)
#include <unordered_set> 

#if (!defined(_WIN32)) && (!defined(_WIN64))
#include <tr1/functional>
#endif

#include <string>
#include <set>
#include <limits>

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX //it messes with limits
#include <windows.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

//Added by Haibo Hao
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <omp.h>
#include "omp.h"
//Added by Haibo Hao

#include "watershedPersistanceAgglomeration.h"
//#include "threadsMylib.h"
#include "graph.h"

//uncomment this to time code and find bottlenecks
#define WATERSHED_TIMING_CPU 

using namespace std;

//structure to save pixel value and position 
//so I can sort foreground elements
struct ForegroundVoxel
{
	//int64: signed long long int. Note by hhb
	int64 pos;
	
	//imgVoxelType:unsigned short int. Note by hhb
	imgVoxelType val; 
        
	//true if the element belongs to the border, 
	//so we have to check more carefully.
	bool isBorder;

	//constructor
	//uint64: unsigned long long int. Note by hhb	
	ForegroundVoxel(uint64 pos_, imgVoxelType val_, 
                  bool isBorder_) : pos(pos_) , 
	          val(val_), isBorder(isBorder_)
	{

	};
	ForegroundVoxel(){ isBorder = false;};

	//operators
	bool operator() (const ForegroundVoxel& lhs,
		       	const ForegroundVoxel& rhs) const
	{
		return lhs.val<rhs.val;
	}

	friend bool operator< ( const ForegroundVoxel& lhs, 
			const ForegroundVoxel& rhs);
};

inline bool operator< (const ForegroundVoxel& lhs,
		 const ForegroundVoxel& rhs)
{
        //believe it or not including this if makes 
        //the sort 8 times slower
	//if( lhs.val == rhs.val) 
	//return lhs.pos > rhs.pos;//so we can get consistent order
	//else
	
	////in order to sort in descending order
	return lhs.val>rhs.val;
};

//====================================================================
//structure to save merges between regions and 
//build the persistance diagram
struct pairPD
{
        //maximum value of the region that is 
        //being merged to another basin with higher fMax
	imgVoxelType fMaxMergedRegion;

        //value at which the merge is happening
	mutable imgVoxelType fMergeVal;

        //position in the image where the maximum of the 
        //region being merged was located (you can use 
        //find(L,fMaxMergedRegionPos) to locate elements)
        
	//labelType:int64. Note by hhb
	labelType fMaxMergedRegionPos;

        //position in the image where the new maximum is located 
        //(it corresponds to the maximum of the region with 
        //highest value of the two being merged)
	labelType fNewMaxRegionPos;

	imgVoxelType getTau()
	{
		return (fMaxMergedRegion - fMergeVal);
	}

	bool operator< (const pairPD& rhs) const
	{
		if( fNewMaxRegionPos == rhs.fNewMaxRegionPos )
			return fMaxMergedRegionPos <
                                 rhs.fMaxMergedRegionPos;
		else
			return fNewMaxRegionPos < rhs.fNewMaxRegionPos;
	}
	bool operator== (const pairPD& rhs) const
	{
		return ( (fNewMaxRegionPos == rhs.fNewMaxRegionPos)
             && (fMaxMergedRegionPos == rhs.fMaxMergedRegionPos) );			
	}
	bool operator() (const pairPD& lhs,const pairPD& rhs) const
	{
		if( lhs.fNewMaxRegionPos == rhs.fNewMaxRegionPos )
			return lhs.fMaxMergedRegionPos < 
				rhs.fMaxMergedRegionPos;
		else
			return lhs.fNewMaxRegionPos < 
				rhs.fNewMaxRegionPos;
	}	
};
//hash function for the struct pairPD
typedef struct
{
          //As a general rule of thumb, if I have two hashes for 
          //independent variables, and I combine them using XOR, 
          //I can expect that the resulting hash is probably just 
          //as good as the input hashes.
	  size_t operator() (const pairPD &p) const 
	  { 
		  //return ( hash<labelType>()(p.fMaxMergedRegionPos
		  //) ^ hash<labelType>()(p.fNewMaxRegionPos) );
		 
#if defined(_WIN32) || defined(_WIN64)
	//from boost: 
	//http://stackoverflow.com/questions/7222143/unordered-map-hash-function-c
	size_t seed = ( hash<labelType>()(p.fMaxMergedRegionPos) )
			 + 0x9e3779b9;
	return ( seed ^ ( ( hash<labelType>()(p.fNewMaxRegionPos) )
		 + 0x9e3779b9 + (seed << 6) + (seed >> 2) ) );

#else
	//from boost: 
	//http://stackoverflow.com/questions/7222143/unordered-map-hash-function-c
	size_t seed = ( std::tr1::hash<labelType>()(
			p.fMaxMergedRegionPos) ) + 0x9e3779b9;
	return ( seed ^ ( ( std::tr1::hash<labelType>()(
		p.fNewMaxRegionPos) ) + 0x9e3779b9  + 
			(seed << 6) + (seed >> 2) ) );

#endif
	  }
} pairPDKeyHash;

//====================================================================
//simple node to calculate bottom-to-top agglomeration dendrogram
//starting from basic regions
struct nodeAgglomeration
{
        //pointer to the node in the dendrogram (i.e. binary tree) 
        //that represents this region
	TreeNode<nodeHierarchicalSegmentation> *nodePtr;
	
	//maximum value of this particular region
	imgVoxelType fMax;

};

//needed to merge nodes
double mergeNodesFunctor(double w1, double w2){ return max(w1,w2);};

GraphEdge* findEdgeWithMinTau(
    		graphFA<nodeAgglomeration>& agglomerateGraph, 
		imgVoxelType& tauE);

GraphEdge* findEdgeWithMinTau(unsigned int e1, 
		graphFA<nodeAgglomeration>& agglomerateGraph, 
		imgVoxelType& tauE);
//the same as bove but for a specific edge
//====================================================================
//needed to pass information to each thread
struct threadWatershedData 
{
	const imgVoxelType* img;
	int64 imgDimsThread[dimsImage];
	uint64 offset;
	imgVoxelType backgroundThr;
	int conn3D; 
	imgVoxelType tau; 
	set_unionC* L; 
	imgLabelType *imgL;

	threadWatershedData(const imgVoxelType* img_,
	  const int64* imgDimsThread_, uint64 offset_,
	  imgVoxelType backgroundThr_, int conn3D_,
          imgVoxelType tau_, set_unionC* L_, imgLabelType *imgL_):
          img(img_), offset(offset_), backgroundThr(backgroundThr_),
          conn3D(conn3D_), tau(tau_),L(L_), imgL(imgL_)
	{
		memcpy(imgDimsThread,imgDimsThread_,
			sizeof(int64) * dimsImage);
	}
};


//===================================================================

// needed to explore the neighborhood around and update values.
// Written as a separate inline function so code is cleaner
inline void insertNeigh(imgVoxelType* neighVal, labelType* neighLabel,
  imgVoxelType* neighFmax, int neighSize, imgVoxelType imgVal, 
  labelType auxLabel, imgVoxelType auxVal, imgVoxelType& fMaxValNeigh,
  labelType& fMaxPosNeigh, imgVoxelType& fMaxVal, labelType& fMaxPos)
{
	neighVal[neighSize] =imgVal;					
	neighLabel[neighSize] =(auxLabel);
	neighFmax[neighSize] = auxVal;

	//find maximum around the neighborhood
	if(neighVal[neighSize] > fMaxValNeigh)//clear maximum
	{
		fMaxValNeigh = neighVal[neighSize];
		fMaxPosNeigh = auxLabel;
	}else if( neighVal[neighSize] == fMaxValNeigh
             && fMaxPosNeigh < 0)
		//plateau and we still have not found
        // a labeled neighborhood in the plateau
	{
		fMaxValNeigh = neighVal[neighSize];
		fMaxPosNeigh = auxLabel;
	}

	//find the maximum around the regions that belong to 
	//the neighborhood				                
	if(auxVal>fMaxVal)
	{
		fMaxVal = auxVal;
		fMaxPos = auxLabel;
	}
}


int watershedPersistanceAgglomeration(const imgVoxelType* img, 
  const int64 *imgDims, imgVoxelType backgroundThr, int conn3D,
  imgVoxelType tau, imgLabelType *imgL, imgLabelType *numLabels, 
  set_unionC* L)
{

#ifdef WATERSHED_TIMING_CPU
	time_t start, end;
#endif

	//set some parameters
	uint64 imgSize = 1;
	for(int ii =0;ii<dimsImage; ii++) imgSize *= imgDims[ii];

	//keep only foreground elements 
    //(in general our images are very sparse)
	vector<ForegroundVoxel> foregroundVec;
	foregroundVec.reserve (imgSize / 3);
	selectForegroundElements(img, imgDims, backgroundThr, conn3D,
			 foregroundVec, imgL);


	//sort elements in P in descending order 
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif
	sort(foregroundVec.begin(), foregroundVec.end());
#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"Sorting took "<<difftime(end,start)<< " secs for "<<foregroundVec.size()<<" voxels"<<endl;
#endif

	//allocate memory for Label ID output
	bool deallocateL = false;
	if(L == NULL)
	{
		//we use union-find data structure to efficiently run algorithm
		L=(set_unionC*)malloc(sizeof(set_unionC));
		//we initialize every object to null assignment
		set_union_init(L,imgSize);
		deallocateL = true;
	}



	//define neighborhood scheme
	//margin we need to calculate connected components
	int64 boundarySize[dimsImage];	
	int64 *neighOffset = buildNeighboorhoodConnectivity(conn3D, 
							imgDims, boundarySize);
	if( neighOffset == NULL)
		return 2;


	//main algorithm to find topologically persistande regions    

	labelType *neighLabel=(labelType*)malloc(sizeof(labelType)*conn3D);
	imgVoxelType *neighVal=(imgVoxelType*)malloc(sizeof(imgVoxelType)*conn3D);
	imgVoxelType *neighFmax=(imgVoxelType*)malloc(sizeof(imgVoxelType)*conn3D);

	int neighSize;//number of elements that have been already labeled

	imgVoxelType Pval = imgVoxelMaxValue, auxVal;
	int64 posB, posBneigh;
	labelType auxLabel;
	imgVoxelType fMaxValNeigh;//maximum value in the neighborhod
	labelType fMaxPosNeigh;
	imgVoxelType fMaxVal;
	labelType fMaxPos;	
	int64 imgCoord[dimsImage];
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif
	for(vector<ForegroundVoxel>::const_iterator iter=foregroundVec.begin();iter!=foregroundVec.end();++iter)
	{
		Pval=iter->val;
		posB=iter->pos;

		neighSize = 0;
		
		//maximum value in the neighborhod
		fMaxValNeigh = imgVoxelMinValue;
		fMaxPosNeigh = -1;
		fMaxVal = imgVoxelMinValue;
		fMaxPos = -1;	

		//copy elements for the neighborhood               
		if( iter->isBorder == false ) // we are not in a border
		{
			for(int jj=0;jj<conn3D;jj++)
			{
				posBneigh = posB + neighOffset[jj];
				//find_and_get_fMax(L,posBneigh,&auxLabel,&auxVal);
				/*
				//this is the bottleneck: it takes 50% of the time
				auxLabel = find(L,posBneigh);
				//we only care about labels that have been assigned 
				//since we have sorted elements
				if(auxLabel >= 0)
				{							
					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,
					   neighSize,  img[posBneigh],  auxLabel, auxVal,  						  fMaxValNeigh, fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;
				}
				*/
				if( img[ posBneigh ] > Pval )
				{	
				 //this is the bottleneck: it takes 50% of the time	
					auxLabel = find(L,posBneigh);

					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,
				      neighSize,  img[posBneigh],  auxLabel,  auxVal, 
                      fMaxValNeigh, fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;					
				}else if(img[posBneigh]==Pval && L->p[posBneigh] >= 0)
				{
					//this is the bottleneck: it takes 50% of the time
					auxLabel = find(L,posBneigh);
					//if( auxLabel >= 0)
					{
						auxVal = L->fMax[auxLabel];
						insertNeigh( neighVal,  neighLabel, neighFmax,
                      neighSize,  img[posBneigh],  auxLabel, auxVal, 
                      fMaxValNeigh, fMaxPosNeigh,  fMaxVal, fMaxPos);
						neighSize++;
					}
				}
			}
		}else{//we are close to a border-> we have to be more cautious

			//calculate coordinates
			int64 auxC = posB;
			for(int aa = 0; aa<dimsImage; aa++)
			{
				imgCoord[aa] = auxC % imgDims[aa];
				auxC -= imgCoord[aa];
				auxC /= imgDims[aa];
			}
			int64 boundaryCounter[dimsImage];
			for(int ii =0;ii<dimsImage;ii++) 
				boundaryCounter[ii] = -boundarySize[ii];
			boundaryCounter[0]--;//to start from "zero"

			for(int jj=0;jj<conn3D;jj++)
			{
				//increment boundary counter
				boundaryCounter[0]++;
				int countC = 0;
				while(boundaryCounter[countC] > boundarySize[countC])
				{
					boundaryCounter[countC] = -boundarySize[countC];
					countC++;
					if(countC>= dimsImage) break;
					boundaryCounter[countC]++;
				}
				countC = 0;
				for(int ii = 0; ii<dimsImage;ii++)
					countC += abs(boundaryCounter[ii]);
				if(countC == 0) 
                  boundaryCounter[0]++;//all zeros is not allowed
				bool isOutOfBounds = false;
				for(int ii = 0;ii<dimsImage;ii++)
				{
					if( imgCoord[ii] + boundaryCounter[ii] < 0)
					{
						isOutOfBounds = true;
						break;
					}
					if( imgCoord[ii] + boundaryCounter[ii] >= imgDims[ii])
					{
						isOutOfBounds = true;
						break;
					}
				}
				if(isOutOfBounds == true) 
					continue;//we should not check this neighbor

				//same for loop as before 
				//(when we were not close to a border)
				posBneigh = posB+neighOffset[jj];
				//out of bounds. TODO: this is not completely correct
				//since we need to check bounds fro x,y,z separately.
				// However, most of our images have background in the
				// borders and it will avoid comparing all the time.
				if(posB < -neighOffset[jj]) 
				{
					posBneigh = 0;
				}else if(posBneigh >= ((int64)(imgSize) ))
				{
					posBneigh = imgSize - 1;
				}
				//find_and_get_fMax(L,posBneigh,&auxLabel,&auxVal);
				/*
				auxLabel = find(L,posBneigh);
				//we only care about labels that have been assigned
				// since we have sorted elements
				if(auxLabel >= 0)
				{	
					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,
				  neighSize,  img[posBneigh],  auxLabel,  auxVal, 
                  fMaxValNeigh, fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;
				}
				*/
				if( img[ posBneigh ] > Pval )
				{						
					//this is the bottleneck: it takes 50% of the time
					auxLabel = find(L,posBneigh);

					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,
                  neighSize,  img[posBneigh],  auxLabel,  auxVal, 
                  fMaxValNeigh, fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;					
				}else if(img[posBneigh]==Pval && L->p[posBneigh] >= 0)
				{
					//this is the bottleneck: it takes 50% of the time
					auxLabel = find(L,posBneigh);

					//if( auxLabel >= 0)
					{
						auxVal = L->fMax[auxLabel];
						insertNeigh( neighVal,  neighLabel, neighFmax,
                     neighSize,  img[posBneigh],  auxLabel,  auxVal, 
                     fMaxValNeigh, fMaxPosNeigh,  fMaxVal, fMaxPos);
						neighSize++;
					}
				}
			}
		}		

		//decide what to do with the value after checking neighbors:
        //3 cases (local maximum, only one label, multiple labels -mayb
        //e merge-) 

		//we have a local maxima or a flat region with no assignments
		//yet
		if(Pval>fMaxValNeigh || (Pval==fMaxValNeigh && fMaxPosNeigh<0))
		{

			add_new_component(L,posB,Pval);//add new component
		}
		else{

			//assign the element to the region with highest value                       

			//COMMENT:VIP:THIS IS A CRITICIAL CHOICE. fMaxPos 
			//GENERERATES BETTER OUTLINES WHILE 0 is REALLY A 
            //PARTITION OF THE SPACE
 
			//TODO: CHECK WHICH ONE WORKS BETTER FOR NUCLEI 
            //SEGMENTATION. A PRIORI, fMaxPos SEEM TO HAVE MORE LEAKAGE

			//merge current pixel to the region with the AVERALL
            //highest value

			//#define MERGE_TO_REGION_FMAX 
#ifdef MERGE_TO_REGION_FMAX
			add_element_to_set(L, fMaxPos, posB,Pval);
#else
			//merge current pixel to the region with the highest
			//value in the neighborhood
			add_element_to_set(L, fMaxPosNeigh, posB,Pval);
#endif

			//check if we have to merge components with lower value
			int numElemToMerge = 0;
			for(int aa = 0; aa < neighSize; aa++)
			{
				//we do not need to check merge with already the
				//highest value
				if(neighLabel[aa] == fMaxPos) 
					continue;

				//merge components using absolute value
				if((neighFmax[aa]-Pval)<=tau)
					//TODO: add a generic "merging decision" function
					// with common inputs so I can play with different
					// strategies

					//if((neighFmax[aa]-Pval)/neighFmax[aa]<=tau)
					//merge components using relative value
				{
					//neighLabelMerge.insert(make_pair(neighLabel[aa],
					//neighFmax[aa]));
					
					//resuign memory
					neighLabel[numElemToMerge] = neighLabel[aa];
					numElemToMerge++;
				}
			}
			for(int aa = 0; aa< numElemToMerge; aa++)
				union_sets(L, fMaxPos, neighLabel[aa]);		

		}

	}
#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"in Watershed  main loop took "<<difftime(end,start)<< " secs"<<endl;
#endif


	if( deallocateL == true)//we are using a single-thread version, so this is the final result
	{
		//parse all the results to output    
		*numLabels = 0;
		unordered_map<labelType,imgLabelType> mapLabel;
		mapLabel.rehash( ceil( (foregroundVec.size() / 50 ) / mapLabel.max_load_factor() ));//to avoid rehash. a.reserve(n) is the ame as a.rehash(ceil(n / a.max_load_factor()))
		unordered_map<labelType,imgLabelType>::const_iterator mapLabelIter;
		//mapLabel.insert( make_pair(1,0) );
		labelType auxL;
		//imgVoxelType auxFmax;
#ifdef WATERSHED_TIMING_CPU
		time(&start);
#endif
		for(vector<ForegroundVoxel>::const_iterator iter=foregroundVec.begin();iter!=foregroundVec.end();++iter)//TODO: it can be faster if we access elements in order in the image for cache friendliness. Right now (iter->pos) jumps all over the image
		{

			auxL = find(L,iter->pos) + 1;//so label 0 is reserved for background
			mapLabelIter = mapLabel.find(auxL);

			if( mapLabelIter == mapLabel.end())//element not found
			{
				//auxFmax = get_fMax(L, iter->pos);
				if(L->fMax[auxL-1] <= tau)//merge components with background
				{
					imgL[iter->pos] = 0;
					mapLabel.insert(make_pair(auxL,0));
				}else{//new label
					if( (*numLabels) >= maxNumLabels)
					{
						cout<<"ERROR: at watershedPersistanceAgglomeration: maximum number of labels "<< maxNumLabels<<"  reached"<<endl;
						return 3;
					}
					(*numLabels)++;
					mapLabel.insert( make_pair(auxL, *numLabels) );				
					imgL [iter->pos] = (*numLabels);
				}
			}else{//element found
				imgL [ iter->pos] = mapLabelIter->second;
			}                
		}

#ifdef WATERSHED_TIMING_CPU
		time(&end);
		cout<<"Parsing labels took "<<difftime(end,start)<< " secs"<<endl;
#endif
	}


	//--------------------------end of second pass------------------------------------------

	/* Free allocated matrices */
	if(deallocateL == true)
	{
		set_union_destroy(L);
		free(L);
	}
	delete[] neighOffset;   
	free(neighLabel);
	free(neighVal);
	free(neighFmax);

	return 0;
};


int watershedPersistanceAgglomerationMultithread(const imgVoxelType* img, const int64 *imgDims, imgVoxelType backgroundThr, int conn3D, imgVoxelType tau, imgLabelType *imgL, imgLabelType *numLabels, int numThreads)
{
#ifdef WATERSHED_TIMING_CPU
	time_t start, end;
#endif

	cout<<"WARNING!!!:at watershedPersistanceAgglomerationMultithread: multithread approach has not been completely tested. Number of final total regions changes +-3% depending on numThreads!!! I need to figure out what is causing that in the merging regions. A good clue are PLATEAUS of intensity"<<endl;

	if(numThreads <= 0)
	{
		numThreads = 8;//TODO: check how many cores are available in the machine
	}

	//set some parameters
	uint64 imgSize = 1;
	for(int ii =0;ii<dimsImage; ii++) imgSize *= imgDims[ii];

	//allocate set_union elements for the whole image
	set_unionC *L=(set_unionC*)malloc(sizeof(set_unionC));//we use union-find data structure to efficiently run algorithm
	set_union_init(L,imgSize);//we initialize every object to null assignment


	//generate the foreground vector and sort it in order to check which regions to merge
	vector<ForegroundVoxel> foregroundVec;
	const int lastDim = dimsImage - 1;
	int64* numForegroundElementsPerSlice = new int64[ imgDims[lastDim] ];//vector in order to distirbute the load between threads equally
	selectForegroundElements(img, imgDims, backgroundThr, conn3D, foregroundVec, imgL, numForegroundElementsPerSlice);
	int64  numForegorundElements = foregroundVec.size();
	//sort elements in P in descending order 
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif
	sort(foregroundVec.begin(), foregroundVec.end());
#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"Sorting before merging watershed threads took "<<difftime(end,start)<< " secs for "<<foregroundVec.size()<<" voxels"<<endl;
#endif

	//calculate watershed in each of the image blocks (we divide along Z for cache friendliness)
	uint64 sliceOffset = imgDims[0];
	for(int ii =1; ii<dimsImage-1;ii++) sliceOffset *=imgDims[ii];
	int64 iniSlice = 0, endSlice = 0, cumSumSlice;//used to calculate beggining and end for each thread 
	uint64 offset = 0;
	int64 imgDimsThread[dimsImage];
	for(int ii =0; ii<dimsImage-1;ii++)
	{
		imgDimsThread[ii] = imgDims[ii];
	}
	

	#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif

	cout<<"ERROR:watershedPersistanceAgglomerationMultithread: not fully tested and made cross-platform compatible"<<endl;
	/*
	pthread_t *tid = new pthread_t[numThreads];      // array of thread IDs
	vector< threadWatershedData* > tidData(numThreads);
	for(int ii =0; ii<numThreads;ii++)//the last thread is special since it picks up all the rest
	{
		//calculate load distribution
		iniSlice = endSlice;
		cumSumSlice = 0;
		while(endSlice < imgDims[lastDim] && cumSumSlice <= (numForegorundElements / numThreads  ) )
		{
			cumSumSlice += numForegroundElementsPerSlice[endSlice];
			endSlice++;
		}
		if( ii == numThreads -1) 
			endSlice = imgDims[lastDim];//to guarantee all slices are processed
		//launch thread
		imgDimsThread[lastDim] = endSlice - iniSlice;
		tidData[ii] = new threadWatershedData(img, imgDimsThread, offset,backgroundThr, conn3D, tau, L, imgL);		
		pthread_create(&(tid[ii]), NULL, &threadWatershedCallback, (void*) (tidData[ii]) );
		//void* err = threadWatershedCallback((void*) (tidData[ii]));
		
		offset += imgDimsThread[lastDim] * sliceOffset;//update offset
	}

	
	//wait for threads to finish (join)
	for (int ii = 0; ii < numThreads; ii++)
	{
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif
		pthread_join(tid[ii], NULL);
		delete tidData[ii];
#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"Join for thread "<<ii<<" took "<<difftime(end,start)<< " secs"<<endl;
#endif
	}
	//delete thread variables
	delete[] tid;
	tidData.clear();
	*/

	#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"Fork/join threadWatershedCallback took "<<difftime(end,start)<< " secs"<<endl;
#endif

	//iterate over regions merging them until no more regions are merged
	vector<labelType> mergePairs;
	mergePairs.reserve(20000);
	mergePairs.push_back(0);//bogus to start iterations

	int numIter = 0;
	while( mergePairs.empty() == false)
	{
#ifdef WATERSHED_TIMING_CPU
		time(&start);
#endif
		//check region merge already does the remapping. Since the last iteration is the one without merges, the remapping is done appropriately
		int err = checkRegionMerge(img, foregroundVec,L, imgDims , conn3D, tau, imgL, numLabels, mergePairs);
		if(err >0)
			return err;

#ifdef WATERSHED_TIMING_CPU
		time(&end);
		cout<<"Merging "<<mergePairs.size()/2<<" regions for the watershed in iteration "<<(numIter++)<<" took "<<difftime(end,start)<< " secs"<<endl;
#endif
	}

	//release memory
	set_union_destroy(L);
	free(L);
	delete[] numForegroundElementsPerSlice;

	return 0;
}

//=======================================================================================================================================
//int threadWatershedCallback(const imgVoxelType* img, const int64* imgDimsThread, uint64 offset,imgVoxelType backgroundThr, int conn3D, imgVoxelType tau, set_unionC* L, imgLabelType *imgL)
void* threadWatershedCallback(void *p)//adapted for pthreads now
{
	#ifdef WATERSHED_TIMING_CPU
		time_t start,end;		
#endif
	//parse info from void pointer
	threadWatershedData* ptr = (threadWatershedData*) p;
	const imgVoxelType* img = ptr->img;
	int64 imgDimsThread[dimsImage];
	for(int ii = 0;ii<dimsImage;ii++) imgDimsThread[ii] = ptr->imgDimsThread[ii];
	uint64 offset = ptr->offset;
	imgVoxelType backgroundThr = ptr->backgroundThr;
	int conn3D = ptr->conn3D; 
	imgVoxelType tau = ptr->tau; 
	set_unionC* L = ptr->L; 
	imgLabelType *imgL = ptr->imgL;

	if( imgDimsThread[ dimsImage -1 ] == 0 ) return NULL;//sometimes we have extra threads

	imgLabelType numLabelsThread;//dummy variable
	uint64 imgSizeThread = imgDimsThread[0];
	for(int jj = 1;jj<dimsImage;jj++) imgSizeThread *= imgDimsThread[jj];

	//alocate memory
	set_unionC *Lthread=(set_unionC*)malloc(sizeof(set_unionC));//we use union-find data structure to efficiently run algorithm
	set_union_init(Lthread,imgSizeThread);

	#ifdef WATERSHED_TIMING_CPU
		time(&start);		
#endif

	//call watershed
	int err = watershedPersistanceAgglomeration(&(img[offset]), imgDimsThread, backgroundThr, conn3D, tau, &(imgL[offset]), &numLabelsThread, Lthread);//TODO: desgined a set_union structure with an offset so can use just a single L for all the threads
	if(err> 0)
		exit( err );

#ifdef WATERSHED_TIMING_CPU
		time(&end);
		cout<<"Watershed for block with offset="<<offset<<" took "<<difftime(end,start)<< " secs"<<endl;
#endif

	#ifdef WATERSHED_TIMING_CPU
		time(&start);
#endif
	//copy Lthread to global L
	uint64 offsetAux = offset;
	for(uint64 ii = 0;ii< imgSizeThread; ii++)
	{
		L->p[offsetAux] = Lthread->p[ii] + offset;
		L->fMax[offsetAux] = Lthread->fMax[ii];
		L->size[offsetAux] = Lthread->size[ii];
		offsetAux++;
	}

	#ifdef WATERSHED_TIMING_CPU
		time(&end);
		cout<<"Copying data for block with offset="<<offset<<" from Lthread to global set_unionC* L took "<<difftime(end,start)<< " secs"<<endl;
#endif

	//free memory
	set_union_destroy(Lthread);
	free(Lthread);

	return NULL;
}

//===============================================================================================================
//I need the foreground all sorted
int checkRegionMerge(const imgVoxelType* img, const vector<ForegroundVoxel>& foregroundVec,set_unionC *L, const int64* imgDims , int conn3D, imgVoxelType tau, imgLabelType* imgL, imgLabelType *numLabels, vector<labelType>& mergePairs)
{
	mergePairs.clear();

	int64 boundarySize[dimsImage];//margin we need to calculate connected components	
	int64 *neighOffset = buildNeighboorhoodConnectivity(conn3D, imgDims, boundarySize);
	if( neighOffset == NULL)
		return 2;

	uint64 imgSize = imgDims[0];
	for(int ii =1;ii<dimsImage; ii++) imgSize *= imgDims[ii];

	imgVoxelType auxFmax, neighFmax;
	labelType auxL, auxNeighLabel;
	uint64 posBneigh;
	int64 posB;

	//I do not need a hash table (a set would eb find) but it tends to be faster for thousands of elements to identify if we have already inserted the element
	unordered_map<labelType,imgLabelType> checkLabels;
	checkLabels.rehash( ceil( (foregroundVec.size() / 50 ) / checkLabels.max_load_factor() ));//to avoid rehash. a.reserve(n) is the ame as a.rehash(ceil(n / a.max_load_factor()))
	checkLabels.insert( make_pair(-1, 0) );
	unordered_map<labelType,imgLabelType>::const_iterator checkLabelsIter;

	(*numLabels) = 0;
	//for(uint64 posB = abs(neighOffset[0]);posB < imgSize - neighOffset[conn3D-1]; posB++) // we skip the initial elementsto avoid out of bounds. TODO: this is not completely clean
	int64 imgCoord[dimsImage];
	for(vector<ForegroundVoxel>::const_iterator iter=foregroundVec.begin();iter!=foregroundVec.end();++iter)
	{
		posB = iter->pos;		
		auxL = find(L,posB);		
		checkLabelsIter = checkLabels.find(auxL);

		if( checkLabelsIter != checkLabels.end())
		{
			imgL[posB] = checkLabelsIter->second;//we do the remapping
			continue;//region has already been investigated
		}
		if( (*numLabels) >= maxNumLabels)
		{
			cout<<"ERROR: at checkRegionMerge: maximum number of labels reached"<<endl;
			return 3;
		}
		(*numLabels) ++;
		checkLabels.insert( make_pair(auxL, (*numLabels) ) );	//insert the new label
		imgL[posB] = (*numLabels);

		//auxL should not be negative since we have check already		
		//auxFmax=get_fMax(L, posB);
		auxFmax = L->fMax[auxL];

		//check all the neighbors
		if(iter->isBorder == false)
		{
			for(int jj=0;jj<conn3D;jj++)
			{
				posBneigh = posB + neighOffset[jj];
				auxNeighLabel = find(L, posBneigh);
				if(auxNeighLabel == auxL || auxL < 0) continue;//they belong to the same region (== relationship) || neighbor is part of background (-1)

				if(img[posBneigh] < iter->val) continue;//neighbor has a lower image value->we should nor have "seen" it yet. PLATEAUS INTRODUCE SMALL DIFFERENCES IN MERGED REGIONS
				//neighFmax=get_fMax(L,auxNeighLabel);
				neighFmax = L->fMax[ auxNeighLabel ];

				if( ( fmin(auxFmax,neighFmax) - iter->val ) <= tau)//they need to be merged
				{
					mergePairs.push_back(auxL);//some pairs would be added multiple times but it is not a big deal. It is also not the bottleneck
					mergePairs.push_back(auxNeighLabel);
				}			
			}
		}else{//we are in a border. We have to be more cautious
			//calculate coordinates
			int64 auxC = posB;
			for(int aa = 0; aa<dimsImage; aa++)
			{
				imgCoord[aa] = auxC % imgDims[aa];
				auxC -= imgCoord[aa];
				auxC /= imgDims[aa];
			}
			int64 boundaryCounter[dimsImage];
			for(int ii =0;ii<dimsImage;ii++) 
				boundaryCounter[ii] = -boundarySize[ii];
			boundaryCounter[0]--;//to start from "zero"
			for(int jj=0;jj<conn3D;jj++)
			{
				//increment boundary counter
				boundaryCounter[0]++;
				int countC = 0;
				while(boundaryCounter[countC] > boundarySize[countC])
				{
					boundaryCounter[countC] = -boundarySize[countC];
					countC++;
					if(countC>= dimsImage) break;
					boundaryCounter[countC]++;
				}
				countC = 0;
				for(int ii = 0; ii<dimsImage;ii++)
					countC += abs(boundaryCounter[ii]);
				if(countC == 0) boundaryCounter[0]++;//all zeros is not allowed
				bool isOutOfBounds = false;
				for(int ii = 0;ii<dimsImage;ii++)
				{
					if( imgCoord[ii] + boundaryCounter[ii] < 0)
					{
						isOutOfBounds = true;
						break;
					}
					if( imgCoord[ii] + boundaryCounter[ii] >= imgDims[ii])
					{
						isOutOfBounds = true;
						break;
					}
				}
				if(isOutOfBounds == true) continue;//we should not check this neighbor

				//same as in the previous for loop
				posBneigh = posB + neighOffset[jj];
				auxNeighLabel = find(L, posBneigh);
				if(auxNeighLabel == auxL || auxL < 0) continue;//they belong to the same region (== relationship) || neighbor is part of background (-1)

				if(img[posBneigh] < iter->val) continue;//neighbor has a lower image value->we should nor have "seen" it yet. PLATEAUS INTRODUCE SMALL DIFFERENCES IN MERGED REGIONS
				//neighFmax=get_fMax(L,auxNeighLabel);
				neighFmax = L->fMax[ auxNeighLabel ];

				if( ( fmin(auxFmax,neighFmax) - iter->val ) <= tau)//they need to be merged
				{
					mergePairs.push_back(auxL);
					mergePairs.push_back(auxNeighLabel);
				}			
			}
		}
	}

	//merge all the pairs
	for(vector<labelType>::iterator iter = mergePairs.begin(); iter != mergePairs.end(); ++iter)
	{
		auxL = *iter;
		iter++;
		//union_sets(L, fMaxPos, neighLabel[aa]);	
		union_sets(L, *iter, auxL);	
	}

	cout<<"NumLabels before merging = "<< *numLabels<<endl;

	//release memory
	delete[] neighOffset;

	return 0;
}

//=====================================================
int64* buildNeighboorhoodConnectivity(int conn3D,
	const int64* imgDims,int64* boundarySize)
 	
{
	int64 *neighOffset= new int64[conn3D];

        //margin we need to calculate connected components
	//int64 boundarySize[dimsImage];

	//VIP: NEIGHOFFSET HAS TO BE IN ORDER
	//(FROM LOWEST OFFSET TO HIGHEST OFFSET IN ORDER 
	//TO CALCULATE FAST IF WE ARE IN A BORDER)
	
	switch(conn3D)
	{
        //4 connected component (so considering neighbors 
	//only in 2D slices)
	case 4:
	    {
  		  neighOffset[1]=-1;
		  neighOffset[2]=1;
		  neighOffset[0]=-imgDims[0];
		  neighOffset[3]=imgDims[0];

		  boundarySize[0] = 1;
		  boundarySize[1] = 1;
		  boundarySize[2] = 0;
		  break;
	    }
        //6 connected components as neighborhood
	case 6:
	    {
 		  neighOffset[2]=-1;
		  neighOffset[3]=1;
		  neighOffset[1]=-imgDims[0];
		  neighOffset[4]=imgDims[0];
		  neighOffset[0]=-imgDims[0]*imgDims[1];
		  neighOffset[5]=imgDims[0]*imgDims[1];         

		  boundarySize[0] = 1;
		  boundarySize[1] = 1;
                  boundarySize[2] = 1;
		  break;
	    }
        //8 connected component (so considering neighbors 
	//only in 2D slices) 
	case 8:
	    {
		  int countW=0;
		  for(int64 yy=-1;yy<=1;yy++)
		  {
		    for(int64 xx=-1;xx<=1;xx++)
		    {
			if(yy==0 && xx==0) 
			  continue;//skipp the middle point

			neighOffset[countW++] = xx+imgDims[0]*yy;
		    }
		  }

		  if(countW!=8)
		  {
		    cout<<"ERROR: at watershedPersistanceAgglomeration: Window size structure has not been completed correctly"<<endl;
				return NULL;
		  }

			boundarySize[0] = 1;
		       	boundarySize[1] = 1;
		       	boundarySize[2] = 0;
			break;
	    }
        //a cube around the pixel in 3D
	case 26:
	    {
	        int countW=0;
	        for(int64 zz=-1;zz<=1;zz++)
	            for(int64 yy=-1;yy<=1;yy++)
	   	        for(int64 xx=-1;xx<=1;xx++)
			    {
			        if(zz==0 && yy==0 && xx==0) 
				    continue;//skipp the middle point

				 neighOffset[countW++] = 
				    xx+imgDims[0]*(yy+imgDims[1]*zz);
			    }
	         if(countW!=26)
		 {
		     cout<<"ERROR: at watershedPersistanceAgglomeration: Window size structure has not been completed correctly"<<endl;
		     return NULL;
		 }

  		 boundarySize[0] = 1;
		 boundarySize[1] = 1;
		 boundarySize[2] = 1;
		 break;
	    }

        //especial case for nuclei in DLSM: [2,2,1] radius windows 
        //to make sure local maxima are not that local
	case 74:
	    {
	      int countW=0;
	      for(int64 zz=-1;zz<=1;zz++)
	         for(int64 yy=-2;yy<=2;yy++)
		     for(int64 xx=-2;xx<=2;xx++)
		     {
		         if(zz==0 && yy==0 && xx==0)
			     continue;//skipp the middle point

			 neighOffset[countW++]=xx+
			     imgDims[0]*(yy+imgDims[1]*zz);
		     }

	      if(countW!=74)
		{
		   cout<<"ERROR: at watershedPersistanceAgglomeration: Window size structure has not been completed correctly"<<endl;
		   return NULL;
		}

	      boundarySize[0] = 2;
	      boundarySize[1] = 2;
 	      boundarySize[2] = 1;
	      break;
	    }

		//especial case to coarsely xplore negihborhood around
		// [+-6,+-3,0] in XY and [+-1,0] in Z. It is also 74 
		//elements but we need to distinguish it from above
	case 75:
	   {
	     int countW=0;
	     for(int64 zz=-1;zz<=1;zz++)
	       for(int64 yy=-6;yy<=6;yy+=3)
	         for(int64 xx=-6;xx<=6;xx+=3)
		 {
		    if(zz==0 && yy==0 && xx==0)
		        continue;//skipp the middle point
		    neighOffset[countW++]=xx+
			imgDims[0]*(yy+imgDims[1]*zz);
		 }
	     if(countW!=74)
	     {
		cout<<"ERROR: at watershedPersistanceAgglomeration: Window size structure has not been completed correctly"<<endl;
		return NULL;
	     }

	     boundarySize[0] = 6; 
	     boundarySize[1] = 6;
	     boundarySize[2] = 1;
	     break;
	   }
	default:
		cout<<"ERROR: at watershedPersistanceAgglomeration: Code not ready for these connected components"<<endl;
		return NULL;
	}

	return neighOffset;
}

//====================================================
void selectForegroundElements(const imgVoxelType* img, 
  const int64* imgDims, imgVoxelType backgroundThr,
  int conn3D,vector<ForegroundVoxel>& foregroundVec,
  imgLabelType* imgL,int64* numForegroundElementsPerSlice)
  
{

	if(dimsImage !=3)
	{
	 //TODO:it might be easy to geenralize to any
	 //full [xx,yy,zz]
	 
	 //but it might be harder to geenralize for 2n
	 //connectivity 
	 
	 //since it is not a nested for loop
		cout<<"ERROR: selectForegroundElements at: code is not ready for this kind or conn3D or this dimensionality. Conn3D = "<<conn3D<<"; dimsImage = "<<dimsImage<<endl;
		exit(3);
	}

	const int lastDim = dimsImage - 1;
	foregroundVec.clear();

	//reset counter
	if(numForegroundElementsPerSlice != NULL)
	    memset(numForegroundElementsPerSlice, 0,
	      sizeof(int64) * imgDims[lastDim]);

        //keeps image coordinates to decide if we are 
	//close to a border 
	int64 imgCoord[dimsImage];
	memset(imgCoord,0,sizeof(uint64) * dimsImage);
	int count;
	bool isBorder;
	int64 boundarySize[dimsImage];
	int64* neighOffset = 
          buildNeighboorhoodConnectivity(conn3D,imgDims,
			  boundarySize);
				

	uint64 imgSize = imgDims[0];
	for(int ii = 1;ii<dimsImage;ii++) 
		imgSize *= imgDims[ii];

	if(imgL != NULL )
     	//reset image label as background
       	  memset(imgL, 0, sizeof(imgLabelType) * imgSize );

	for(uint64 ii=0;ii<imgSize;ii++)
	{
          if(img[ii] > backgroundThr)
	  {
            //decide if we are close to a border
	    isBorder = false;
     	    for(int jj =0; jj<dimsImage; jj++)
	    {
  	      if(imgCoord[jj]<boundarySize[jj] ||
	        imgCoord[jj]>=imgDims[jj]-boundarySize[jj])
		{
		  isBorder = true;
		  break;
		}
	    }
	    //save values
	    foregroundVec.push_back( 
    	      ForegroundVoxel(ii, img[ii], isBorder));
    	    if(numForegroundElementsPerSlice != NULL)
	      numForegroundElementsPerSlice[
		         imgCoord[lastDim]]++;
	   }

	   //upgrade counter for coordinates
	   count = 0;
	   imgCoord[0]++;
	   while( imgCoord[count] >= imgDims[count])
	   {
		imgCoord[count] = 0;
		count++;
		if(count >= dimsImage) break;
		imgCoord[count]++;
	   }
	}
	delete[] neighOffset;
}


//==============================================================================================================================
int averageNumberOfNeighborsPerRegion(const imgLabelType* imgL,const int64* imgDims, int conn3D, vector<float> &avgNumberOfNeighbors)
{
	avgNumberOfNeighbors.resize(maxNumLabels);//resize
	memset(&(avgNumberOfNeighbors[0]), 0, sizeof(float) * avgNumberOfNeighbors.size() );

	//generate offset for conn3D
	int64 boundarySize[dimsImage];//margin we need to calculate connected components	
	int64 *neighOffset = buildNeighboorhoodConnectivity(conn3D, imgDims, boundarySize);
	if( neighOffset == NULL)
		return 2;



	//we only check within boundaries
	imgLabelType aux;
	int64 posNeigh;
	vector<int> area(avgNumberOfNeighbors.size(), 0);
	imgLabelType numLabels = 0;
	for(int64 zz = boundarySize[2]; zz < imgDims[2] - boundarySize[2]; zz++)
	{
		int64 pos1 = zz * imgDims[0] * imgDims[1];
		for(int64 yy = boundarySize[1]; yy < imgDims[1] - boundarySize[1]; yy++)
		{
			int64 pos2 = pos1 + yy * imgDims[0];
			for(int64 xx = boundarySize[0]; xx < imgDims[0] - boundarySize[0]; xx++)
			{
				aux = imgL[pos2];
				
				if(aux > 0)//not background->update
				{
					int auxN = 0;
					for(int jj=0;jj<conn3D;jj++)
					{
						posNeigh = pos2 + neighOffset[jj];
						
						if( imgL[posNeigh] == aux )//correct neighbor
						{
							auxN++;
						}
					}
					avgNumberOfNeighbors[aux] += ( (float) auxN);
					area[aux]++;
					numLabels = max(numLabels,aux);
				}

				pos2++;
			}
		}
	}
	numLabels++;//because we count 0 (background) as a label

	//calculate average
	avgNumberOfNeighbors.resize(numLabels);
	for(imgLabelType ii = 0; ii < numLabels;ii++)
	{
		if(area[ii] > 0 )
			avgNumberOfNeighbors[ii] /= ( (float) area[ii] );
	}
	


	//release memory
	delete[] neighOffset;
	area.clear();


	return 0;
}

//=======================================================

hierarchicalSegmentation* buildHierarchicalSegmentation(
  		imgVoxelType *img, const int64 *imgDims,
  		imgVoxelType backgroundThr, int conn3D, 
  		imgVoxelType minTau, int numThreads)
{

#ifdef WATERSHED_TIMING_CPU
	time_t buildH_start, buildH_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	time(&buildH_start);
#endif
 

#ifdef WATERSHED_TIMING_CPU
	time_t start, end;
#endif

	//set some parameters
	uint64 imgSize = 1;
	for(int ii =0;ii<dimsImage; ii++) 
		imgSize *= imgDims[ii];

	//keep only foreground elements
	//(in general our images are very sparse)
	vector<ForegroundVoxel> foregroundVec;
	foregroundVec.reserve (imgSize / 3);
	selectForegroundElements(img,imgDims,backgroundThr,
		       	conn3D, foregroundVec, NULL, NULL);


	//sort elements in P in descending order 
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif
	sort(foregroundVec.begin(), foregroundVec.end());
#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"in buildH Sorting took "<<
	    difftime(end,start)<<
	    " secs for "<<foregroundVec.size()<<
	    " voxels"<<endl;
#endif

	//allocate memory for Label ID output
	
	//we use union-find data structure to efficiently
	//run algorithm
	set_unionC* L=(set_unionC*)malloc(sizeof(set_unionC));
	//we initialize every object to null assignment
	set_union_init(L,imgSize);
	
	//allocate memory for persistance diagram
	//vector< pairPD > PDvec;//store unique elements
	//PDvec.reserve(foregroundVec.size());
	//set< pairPD > PDvec;
	
	//checks if an element is unique or not (hash table 
	//look up in average is constant time isntead of 
	//logarithmic)
	unordered_set< pairPD, pairPDKeyHash> PDhash;

	//checks if an element is unique or not (hash table
	//look up in average is constant time isntead of
	//logarithmic)
	pair< unordered_set< pairPD, pairPDKeyHash>::iterator,
	       	bool > PDhashIter;

	//to avoid rehash. a.reserve(n) is the same as 
	//a.rehash(ceil(n / a.max_load_factor()))
	PDhash.rehash(ceil( 
	 (foregroundVec.size()/50)/PDhash.max_load_factor()));
	//PDvec.reserve( foregroundVec.size() / 50  );

	//define neighborhood scheme
	//margin we need to calculate connected components
	int64 boundarySize[dimsImage];	
	int64 *neighOffset = buildNeighboorhoodConnectivity(
			conn3D, imgDims, boundarySize);
	if(neighOffset == NULL)
		return NULL;


	//main algorithm to find topologically persistant regions    
	labelType *neighLabel=(labelType*)malloc(
				sizeof(labelType)*conn3D);
	imgVoxelType *neighVal=(imgVoxelType*)malloc(
				sizeof(imgVoxelType)*conn3D);
	imgVoxelType *neighFmax=(imgVoxelType*)malloc(
				sizeof(imgVoxelType)*conn3D);



//#define USE_MULTITHREAD: much slower because of the mutex

#ifdef USE_MULTITHREAD
	//for multithread purposes
	labelType *neighLabelMT=(labelType*)malloc(
				  sizeof(labelType)*conn3D);
	imgVoxelType *neighValMT=(imgVoxelType*)malloc(
			           sizeof(imgVoxelType)*conn3D);
	imgVoxelType *neighFmaxMT=(imgVoxelType*)malloc(
			            sizeof(imgVoxelType)*conn3D);


	//set main data structure
	threadNeighborhoodInfoData::img = img;
	threadNeighborhoodInfoData::L = L;
	threadNeighborhoodInfoData::neighFmaxMT = neighFmaxMT;
	threadNeighborhoodInfoData::neighValMT = neighValMT;
	threadNeighborhoodInfoData:: neighLabelMT = neighLabelMT;
	threadNeighborhoodInfoData::neighOffset = neighOffset;
	threadNeighborhoodInfoData::conn3D = conn3D;

	threadNeighborhoodInfoData::hEvent = CreateEvent(
			NULL, // No security attributes
                        FALSE, // Automatic-reset event
                        FALSE, // Initial state is not signaled
                        NULL);

	//initialize conition varibales
	InitializeCriticalSection(
		    &(threadNeighborhoodInfoData::BufferLock) );
	InitializeConditionVariable(
	 	   &(threadNeighborhoodInfoData::BufferNotEmpty) );
	threadNeighborhoodInfoData::queueSize = -1;
	threadNeighborhoodInfoData::jobsDone = 0;
	
	
	//create threads
	threadNeighborhoodInfoData* threadData =
	       	new threadNeighborhoodInfoData[numThreads];
	vector< HANDLE > threadHandle(numThreads);
	for(int jj = 0; jj < numThreads; jj++)
	{
 	   threadHandle[jj] = CreateThread(
	     NULL, 
	     0,
	     (LPTHREAD_START_ROUTINE)&(threadNeighborhoodInfoProc),
	     (LPVOID) (&(threadData[jj])),
	     0,
	     NULL);

	   if(threadHandle[jj] == NULL )
	   {
		cout<<"ERROR: creating thread"<<endl;
		exit(3);
	   }
	}
#endif
	//number of elements that have been already labeled
	int neighSize;

	imgVoxelType Pval = imgVoxelMaxValue, auxVal;
	int64 posB, posBneigh;
	labelType auxLabel;
	//maximum value in the neighborhood
	imgVoxelType fMaxValNeigh;
	labelType fMaxPosNeigh;
	imgVoxelType fMaxVal;
	labelType fMaxPos;	
	int64 imgCoord[dimsImage];
	pairPD PDaux;
	int numLabels = 0;
	//to avoid adding to PDvec redundant elements
	vector<labelType> PDmergedElem;
	PDmergedElem.reserve( conn3D );

	//stores unique labels that need to be merged.
	//It is more efficient since we do not need to 
	//call union_sets unnecessarily
	vector<labelType> elemToMerge;
	
	//vector with linear find is faster than set for small arrays
	elemToMerge.reserve( conn3D );

	pair< set<pairPD>::iterator, bool> PDvecIter;

#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif	

	int countDebug = 0;
	for(vector<ForegroundVoxel>::const_iterator iter=foregroundVec.begin();
        iter!=foregroundVec.end();++iter, ++countDebug)
	{
		Pval=iter->val;
		posB=iter->pos;

		neighSize = 0;

		//maximum value in the neighborhod
		fMaxValNeigh = imgVoxelMinValue;
		//position of the maximum value in the neighboorhood
		fMaxPosNeigh = -1; 
		fMaxVal = imgVoxelMinValue;
		fMaxPos = -1;	

		//copy elements for the neighborhood   
		if( iter->isBorder == false ) // we are not in a border
		{

#ifndef USE_MULTITHREAD
	 	  for(int jj=0;jj<conn3D;jj++)
		  {
		      posBneigh = posB + neighOffset[jj];
		      //find_and_get_fMax(L,posBneigh,&auxLabel,&auxVal);
		      
		      //this is the bottleneck: it takes 50% of the time
		      //auxLabel = find(L,posBneigh);
			
	     	  //we only care about labels that have been 
		      //assigned since we have sorted elements
		      //if(auxLabel >= 0)
				if( img[ posBneigh ] > Pval )
				{	
                    //this is the bottleneck: it takes 50% of the time					
					auxLabel = find(L,posBneigh);

					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,neighSize,  
                            img[posBneigh],  auxLabel,  auxVal,  fMaxValNeigh, 
                            fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;					
				}
				else if( img[ posBneigh ] == Pval && L->p[posBneigh] >= 0)
				{  
                    //this is the bottleneck: it takes 50% of the time
					auxLabel = find(L,posBneigh);
					//if( auxLabel >= 0)
					{
						auxVal = L->fMax[auxLabel];
						insertNeigh( neighVal,  neighLabel, neighFmax,neighSize, 
                                img[posBneigh],  auxLabel,  auxVal,  fMaxValNeigh,
                                fMaxPosNeigh,  fMaxVal, fMaxPos);
						neighSize++;
					}
				}
			}
#else

			EnterCriticalSection(&(threadNeighborhoodInfoData::BufferLock));//mutex
			threadNeighborhoodInfoData::imgVal = Pval;
			threadNeighborhoodInfoData::posVal = posB;
			threadNeighborhoodInfoData::jobsDone = 0;
			threadNeighborhoodInfoData::queueSize = conn3D - 1;//use as index
			//if( countDebug % 10000 == 0)
			//	std::cout<<"Main function just updated QueueSize="<<threadNeighborhoodInfoData::queueSize<<". Voxel "<<countDebug<<" out of "<<foregroundVec.size()<<endl;
			LeaveCriticalSection(&(threadNeighborhoodInfoData::BufferLock));//mutex

			//cout<<"Main function waking up all"<<endl;
			WakeConditionVariable( &(threadNeighborhoodInfoData::BufferNotEmpty ) );//signal all the threads they can work

			//wait for signal saying we are done
			WaitForSingleObject(threadNeighborhoodInfoData::hEvent, INFINITE);


			/*
			for(int jj=0;jj<conn3D;jj++)
			{
				cout<<"jj="<<jj<<" "<<threadData[jj].neighLabelMT[jj]<<"=?";
				neighborhoodInfo((void*) (threadData + jj));//&(threadData[jj])
				cout<<"jj="<<jj<<" "<<threadData[jj].neighLabelMT[jj]<<endl;
			}
			*/

			//join information from all threads
			for(int jj=0;jj<conn3D;jj++)
			{
				if( neighFmaxMT[jj] > imgVoxelMinValue )//valid region
				{
					//find maximum around the neighborhood
					if(neighValMT[jj] > fMaxValNeigh)//clear maximum
					{
						fMaxValNeigh = neighValMT[jj];
						fMaxPosNeigh = neighLabelMT[jj];
					}else if( neighValMT[jj] == fMaxValNeigh && fMaxPosNeigh < 0)//plateau and we still have not found a labeled neighborhood in the plateau
					{
						fMaxValNeigh = neighValMT[jj];
						fMaxPosNeigh = neighLabelMT[jj];
					}

					//find the maximum around the regions that belong to the neighborhood				                
					if(neighFmaxMT[jj]>fMaxVal)
					{
						fMaxVal = neighFmaxMT[jj];
						fMaxPos = neighLabelMT[jj];
					}
					//copy elements
					neighVal[neighSize] = neighValMT[jj];
					neighLabel[neighSize] = neighLabelMT[jj];
					neighFmax[neighSize] = neighFmaxMT[jj];

					neighSize++;
				}
			}
			
#endif
		}else{//we are close to a border-> we have to be more cautious

			//calculate coordinates
			int64 auxC = posB;
			for(int aa = 0; aa<dimsImage; aa++)
			{
				imgCoord[aa] = auxC % imgDims[aa];
				auxC -= imgCoord[aa];
				auxC /= imgDims[aa];
			}
			int64 boundaryCounter[dimsImage];
			for(int ii =0;ii<dimsImage;ii++) 
				boundaryCounter[ii] = -boundarySize[ii];
			boundaryCounter[0]--;//to start from "zero"

			for(int jj=0;jj<conn3D;jj++)
			{
				//increment boundary counter
				boundaryCounter[0]++;
				int countC = 0;
				while(boundaryCounter[countC] > boundarySize[countC])
				{
					boundaryCounter[countC] = -boundarySize[countC];
					countC++;
					if(countC>= dimsImage)
                        break;
					boundaryCounter[countC]++;
				}
				countC = 0;
				for(int ii = 0; ii<dimsImage;ii++)
					countC += abs(boundaryCounter[ii]);
                
                //all zeros is not allowed
				if(countC == 0) 
                    boundaryCounter[0]++;

				bool isOutOfBounds = false;
				for(int ii = 0;ii<dimsImage;ii++)
				{
					if( imgCoord[ii] + boundaryCounter[ii] < 0)
					{
						isOutOfBounds = true;
						break;
					}
					if( imgCoord[ii] + boundaryCounter[ii] >= imgDims[ii])
					{
						isOutOfBounds = true;
						break;
					}
				}
				if(isOutOfBounds == true) 
					continue;//we should not check this neighbor

				//same for loop as before (when we were not close to a border)
				posBneigh = posB+neighOffset[jj];

                //out of bounds. TODO: this is not completely correct
                //since we need to check bounds from x,y,z separately.
                //However, most of our images have background in the
                //borders and it will avoid comparing all the time. 
				if(posB < -neighOffset[jj]) 
				{
					posBneigh = 0;
				}else if(posBneigh >= ((int64)(imgSize) ))
				{
					posBneigh = imgSize - 1;
				}
				//find_and_get_fMax(L,posBneigh,&auxLabel,&auxVal);
				/*
				auxLabel = find(L,posBneigh);

                //we only care about labels that have been assigned 
                //since we have sorted elements
				if(auxLabel >= 0)
				{	
					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,neighSize,
                         img[posBneigh],  auxLabel,  auxVal,  fMaxValNeigh, 
                         fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;
				}
				*/
				if( img[ posBneigh ] > Pval )
				{						
                    //this is the bottleneck: it takes 50% of the time 
					auxLabel = find(L,posBneigh);
					auxVal = L->fMax[auxLabel];
					insertNeigh( neighVal,  neighLabel, neighFmax,neighSize,
                            img[posBneigh],  auxLabel,  auxVal,  fMaxValNeigh,
                            fMaxPosNeigh,  fMaxVal, fMaxPos);
					neighSize++;					
				}else if( img[ posBneigh ] == Pval && L->p[posBneigh] >= 0)
				{
                    //this is the bottleneck: it takes 50% of the time
					auxLabel = find(L,posBneigh);
					//if( auxLabel >= 0)
					{
						auxVal = L->fMax[auxLabel];
						insertNeigh( neighVal,  neighLabel, neighFmax,neighSize,
                                img[posBneigh],  auxLabel,  auxVal,  fMaxValNeigh,
                                fMaxPosNeigh,  fMaxVal, fMaxPos);
						neighSize++;
					}
				}
			}
		}		

		//decide what to do with the value after checking neighbors:
        //3 cases (local maximum, only one label, multiple labels -maybe merge-) 
        
        //we have a local maxima or a flat region with no assignments yet
		if(Pval>fMaxValNeigh || (Pval == fMaxValNeigh && fMaxPosNeigh < 0) )
		{

			add_new_component(L,posB,Pval);//add new component
			numLabels++;
		}
		else{

			//assign the element to the region with highest value                       

			//COMMENT:VIP:THIS IS A CRITICIAL CHOICE. fMaxPos GENERERATES 
            //BETTER OUTLINES WHILE 0 is REALLY A PARTITION OF THE SPACE
            
			//TODO: CHECK WHICH ONE WORKS BETTER FOR NUCLEI SEGMENTATION.
            //A PRIORI, fMaxPos SEEM TO HAVE MORE LEAKAGE
            
			//merge current pixel to the region with the AVERALL highest value
            
			//#define MERGE_TO_REGION_FMAX 
#ifdef MERGE_TO_REGION_FMAX
			add_element_to_set(L, fMaxPos, posB,Pval);
#else
			//merge current pixel to the region with the highest value 
            //in the neighborhood
			add_element_to_set(L, fMaxPosNeigh, posB,Pval);
#endif

			//check if we have to merge components with lower value			
			elemToMerge.clear();
			vector<labelType>::iterator ret;
			PDmergedElem.clear();
			for(int aa = 0; aa < neighSize; aa++)
			{
                //we do not need to check merge since they belong 
                //already to maximum value
				if(neighLabel[aa] == fMaxPos) 
					continue;								

                //element neighLabel[aa] was already checked
				if( find(PDmergedElem.begin(), PDmergedElem.end(), neighLabel[aa] )
                        != PDmergedElem.end() )
					continue;

				PDmergedElem.push_back(neighLabel[aa]);


                //merge components using absolute value
                
                //TODO: add a generic "merging decision" function
                //with common inputs so I can play with different strategies/
				if((neighFmax[aa]-Pval)<=minTau)					
					//if((neighFmax[aa]-Pval)/neighFmax[aa]<=tau)
                    //merge components using relative value
				{
					if( find( elemToMerge.begin(), elemToMerge.end(), neighLabel[aa] ) 
                            == elemToMerge.end() )//element was not inserted
						elemToMerge.push_back(neighLabel[aa] );
				}else{
					//to build PD we would run the code with tau = Inf 
                    //( so basically we would merge everything, and everytime
                    //we merge something we output the pair (neighFmax[aa], Pval)
					PDaux.fMaxMergedRegion = neighFmax[aa];
					PDaux.fMergeVal = Pval;
					PDaux.fMaxMergedRegionPos = neighLabel[aa];
					PDaux.fNewMaxRegionPos = fMaxPos;
					
					//check if element has already been inserted
					PDhashIter = PDhash.insert( PDaux );
					if( PDhashIter.second == false )//update element
					{
						PDhashIter.first->fMergeVal = 
                            std::max( PDhashIter.first->fMergeVal, Pval );
					}														
				}
			}

			for(vector<labelType>::iterator iter = elemToMerge.begin(); 
                    iter != elemToMerge.end(); ++iter)
			{
				union_sets(L, fMaxPos, *iter);		
				numLabels--;
			}

		}

	}
#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"in buildH main loop took "<<difftime(end,start)<< " secs"<<endl;
#endif



	//parse results to hierarchical structure
	//first, basic regions
	hierarchicalSegmentation* hs = new hierarchicalSegmentation(numLabels);

	uint64 imgDimsUINT64[dimsImage];
	for(int ii = 0;ii <dimsImage; ii++)
		imgDimsUINT64[ii] = imgDims[ii];
	supervoxel::setDataDims(imgDimsUINT64);
	supervoxel::dataSizeInBytes = sizeof(imgVoxelType) * imgSize;


	unordered_map<labelType,imgLabelType> mapLabel;
	mapLabel.rehash( ceil( (foregroundVec.size() / 50 ) / mapLabel.max_load_factor() ));//to avoid rehash. a.reserve(n) is the ame as a.rehash(ceil(n / a.max_load_factor()))
	unordered_map<labelType,imgLabelType>::const_iterator mapLabelIter;
	//mapLabel.insert( make_pair(1,0) );
	labelType auxL;
	int numBasicRegions = 0;
	//imgVoxelType auxFmax;
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif

	for(vector<ForegroundVoxel>::const_iterator iter=foregroundVec.begin();iter!=foregroundVec.end();++iter)//TODO: it can be faster if we access elements in order in the image for cache friendliness. Right now (iter->pos) jumps all over the image
	{

		auxL = find(L,iter->pos);//here we do not need to reserve label 0 for background
		mapLabelIter = mapLabel.find(auxL);

		if( mapLabelIter == mapLabel.end())//element not found
		{
			//auxFmax = get_fMax(L, iter->pos);
			if(L->fMax[auxL] > minTau)//if L->fMax[auxL] < minTau, we consider region as part of the background
			{
				//new label
				if( numBasicRegions >= numLabels)
				{
					cout<<"ERROR: at buildHierarchicalSegmentation: maximum number of labels "<< numLabels<<"  reached"<<endl;
					delete hs;
					return NULL;
				}
				mapLabel.insert( make_pair(auxL, numBasicRegions) );				
				//imgL [iter->pos] = (*numLabels);
				hs->basicRegionsVec[ numBasicRegions ].PixelIdxList.push_back( iter->pos );
				hs->basicRegionsVec[ numBasicRegions ].tempWildcard = L->fMax[auxL];//we store the maximum value of this basic region
				hs->basicRegionsVec[ numBasicRegions ].dataPtr = (void*)(img);
				numBasicRegions++;
			}
		}else{//element found
			//imgL [ iter->pos] = mapLabelIter->second;
			hs->basicRegionsVec[ mapLabelIter->second ].PixelIdxList.push_back( iter->pos );
		}    
	}

	int err = hs->shrinkBasicRegionsVec(numBasicRegions);
	if( err > 0)
	{
		delete hs;
		return NULL;
	}

	//sort PixelIdxList for each basic region
	for(unsigned int ii = 0; ii < hs->getNumberOfBasicRegions(); ii++)
	{
		sort(hs->basicRegionsVec[ii].PixelIdxList.begin(),hs->basicRegionsVec[ii].PixelIdxList.end());
	}

#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"Parsing basic regions took "<<difftime(end,start)<< " secs"<<endl;
#endif


	//cout<<"DEBUGGING: number of basic regions "<<numBasicRegions<<"; number of PDvec points "<<PDhash.size()<<endl;

	//second, build dendrogram from basic regions
#ifdef WATERSHED_TIMING_CPU
	time(&start);
#endif

	//build dendrogram (it is basically an agglomeration algorithm). Check notebook February 5th 2013
	graphFA<nodeAgglomeration> agglomerateGraph(false);//unidrected graph. We have to merge all the edges until only one node stands
	agglomerateGraph.reserveNodeSpace( hs->getNumberOfBasicRegions() );

	//initialize node agglomeration: bottom of the binary tree containing basic regions
	nodeAgglomeration auxNode;
	for(unsigned int countVec = 0; countVec <  hs->getNumberOfBasicRegions(); countVec++)
	{
		auxNode.fMax = ((imgVoxelType) hs->basicRegionsVec[countVec].tempWildcard );
		auxNode.nodePtr = new TreeNode<nodeHierarchicalSegmentation>();
		auxNode.nodePtr->left = NULL;
		auxNode.nodePtr->right = NULL;
		auxNode.nodePtr->parent = NULL;
		auxNode.nodePtr->data.svPtr = &(hs->basicRegionsVec[countVec]);
		auxNode.nodePtr->data.thrTau = minTau;
		//add it to the graph
		agglomerateGraph.insert_node( auxNode );
	}	

	

	//cout<<"DEBUGGING:parse PD into a set of unique edges between basic regions (nodes in the graph) to initialize all the edges in agglomerationGraph"<<endl;
	labelType id1, id2;
	GraphEdge* auxEdge;

	int numSelfEdges = 0;
	for(unordered_set< pairPD, pairPDKeyHash >::iterator iterPD = PDhash.begin(); iterPD!= PDhash.end(); ++iterPD)
	{
		auxL = find(L, iterPD->fNewMaxRegionPos);
		id1 = mapLabel.find(auxL)->second;
		auxL = find(L, iterPD->fMaxMergedRegionPos);		
		id2 = mapLabel.find(auxL)->second;
		
		if( id1 == id2)
		{
			numSelfEdges++;
			continue;//TODO: this should not happen, but in some very large images it was crashing the code (since we get self edges). For example, in mouse 12-08-2010_TM0, 34 / 675393 nodes have self edges.
		}
		//check if edge has already been stablished
		auxEdge = agglomerateGraph.findEdge(id1, id2);

		if ( auxEdge == NULL )//not found->insert new edge
		{
			agglomerateGraph.insert_edge(id1,id2,iterPD->fMergeVal);//the weight is the value at which both regions touch
		}else{//found->update edge
			auxEdge->weight = std::max( auxEdge->weight, (double)(iterPD->fMergeVal));//the weight is the value at which both regions touch (we are looking for maxima)
		}

	}

	if( numSelfEdges > 0 )
	{
		cout<<"WARNING: buildHierarchicalSegmentation: num nodes with self edges = "<<numSelfEdges<<" out of "<<agglomerateGraph.getNumNodes_Omp()<<". Right now I have a quick fix but this should not happen"<<endl;
	}

	
	//cout<<"DEBUGGING:start agglomeration loop"<<endl;
	unsigned int e1, e2;
	nodeAgglomeration e1NodeOld;
	vector<GraphEdge*> e1EdgesOld;
	double (*mergeNodesFunctorPtr)(double, double);
	mergeNodesFunctorPtr = &mergeNodesFunctor;


	//to find edgeWithMinTau efficiently
	vector< pair<GraphEdge*,imgVoxelType> > minTauVec( agglomerateGraph.nodes.size() );
	vector< unsigned int > updateEdgeVec;
	updateEdgeVec.reserve(50);

	//initialize
	unsigned int countE = 0;
	for(vector< pair<GraphEdge*,imgVoxelType> >::iterator iter = minTauVec.begin(); iter!= minTauVec.end(); ++iter, ++countE)
	{
		iter->first = findEdgeWithMinTau(countE, agglomerateGraph, iter->second );
	}


	//--------------------------------------------------------------------
	/*
	cout<<"DEBUGGING: find edges that have a self-edge"<<endl;
	GraphEdge* auxD = NULL;
	for(size_t nn = 0; nn < agglomerateGraph.getNumNodes(); nn++)
	{
		auxD = agglomerateGraph.findEdge(nn, nn);
		if(  auxD != NULL )
		{
			cout<<"Node "<<nn<<" has a self edge with weight "<<auxD->weight<<endl;
		}
	}
	exit(3);
	*/
	//-------------------------------------------------------------------


	int iterD = 0;
	size_t numNodes = agglomerateGraph.getNumNodes_Omp();
#ifdef WATERSHED_TIMING_CPU
	double while1_start, while1_end;
#endif
	//double while1_start, while1_end;
	//while1_start = omp_get_wtime();
#ifdef WATERSHED_TIMING_CPU
	while1_start = omp_get_wtime();
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_for1_start, while1_for1_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_for2_start, while1_for2_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_while1_start, while1_while1_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_while2_start, while1_while2_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_sort_start, while1_sort_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_phase1_start, while1_phase1_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	double  while1_phase2_start, while1_phase2_end;
#endif

#ifdef WATERSHED_TIMING_CPU
   double  while1_phase2_p1_start, while1_phase2_p1_end;
   double  while1_phase2_p2_start, while1_phase2_p2_end;
   double  while1_phase2_p3_start, while1_phase2_p3_end;
#endif
 
long long int count_while1 = 0;

double sum_time_while1 = 0;
double sum_time_while1_for1 = 0;
double sum_time_while1_for2 = 0;
double sum_time_while1_while1 = 0;
double sum_time_while1_while2 = 0;
double sum_time_while1_sort = 0;
double sum_time_while1_phase1 = 0;
double sum_time_while1_phase2 = 0;
double sum_time_while1_phase2_p1 = 0;
double sum_time_while1_phase2_p2 = 0;
double sum_time_while1_phase2_p3 = 0;

	while( numNodes > 1 )
	{

#ifdef WATERSHED_TIMING_CPU
	while1_phase1_start = omp_get_wtime();
#endif
		count_while1++;
		//find node with min tau (next merging)
		//imgVoxelType tauE;
		//GraphEdge* edgeMin = findEdgeWithMinTau(agglomerateGraph, tauE); //this function is too slow since it checks all teh edges everytime

		//find the minimum across all nodes
		GraphEdge* edgeMin = NULL;
		imgVoxelType tauE = imgVoxelMaxValue;

#ifdef WATERSHED_TIMING_CPU
	while1_for1_start = omp_get_wtime();	
#endif
		for(vector< pair<GraphEdge*,imgVoxelType> >::iterator iter = minTauVec.begin(); iter!= minTauVec.end(); ++iter)
		{
			if( iter->second < tauE )
			{
				tauE = iter->second;
				edgeMin = iter->first;
			}
		}

#ifdef WATERSHED_TIMING_CPU
	while1_for1_end = omp_get_wtime();

	sum_time_while1_for1 += 
	 (double)(while1_for1_end-
	  while1_for1_start);
#endif
		
		if(edgeMin == NULL)
		{
			//it means we have disconnected completely diconnected regions
			break;
		}
		e1 = edgeMin->e1;
		e2 = edgeMin->e2;
		e1NodeOld = agglomerateGraph.nodes[ e1 ];
		
		//check that agglomeration was correct
		if( e1 == e2 )
		{
			cout<<"ERROR: buildHierarchicalSegmentation: e1 and e2 are the same node"<<endl;
			cout<<"Number of edges left in e1 = "<<agglomerateGraph.getNumEdges(e1)<<endl;
			cout<<"Check point 2: iter="<<iterD<<";numNodes = "<<numNodes<<".Selected nodes:e1="<<e1<<";e2="<<e2<<";tauE="<<tauE<<endl;			
			exit(3);
		}

		//cout<<"Check point 2: iter="<<(iterD)<<";numNodes = "<<numNodes<<".Selected nodes:e1="<<e1<<";e2="<<e2<<";tauE="<<tauE<<endl;

		//create new node in the binary tree 
		auxNode.fMax = max(e1NodeOld.fMax, agglomerateGraph.nodes[ e2 ].fMax);
		auxNode.nodePtr = new TreeNode<nodeHierarchicalSegmentation>();
		auxNode.nodePtr->left = e1NodeOld.nodePtr;
		auxNode.nodePtr->right = agglomerateGraph.nodes[ e2 ].nodePtr;
		auxNode.nodePtr->parent = NULL;
		auxNode.nodePtr->data.svPtr = NULL;//it is not a basic region anymore
		auxNode.nodePtr->data.thrTau = tauE;

		//update parents for e1 and e2
		e1NodeOld.nodePtr->parent = auxNode.nodePtr;
		agglomerateGraph.nodes[ e2 ].nodePtr->parent = auxNode.nodePtr;
		

		

		//save a list of the nodes where I have to update min tau (I use teh fact that graph is bidirectional)
		updateEdgeVec.clear();
		updateEdgeVec.push_back(e1);
		edgeMin = agglomerateGraph.edges[e1];

#ifdef WATERSHED_TIMING_CPU
	while1_while1_start = omp_get_wtime();
#endif
		while( edgeMin != NULL )
		{
		   updateEdgeVec.push_back( edgeMin->e2 );
		   edgeMin = edgeMin->next;
		}

#ifdef WATERSHED_TIMING_CPU
   while1_while1_end = omp_get_wtime();	
   sum_time_while1_while1 +=  
      (double)(while1_while1_end - 
	while1_while1_start);
#endif
		edgeMin = agglomerateGraph.edges[e2];

#ifdef WATERSHED_TIMING_CPU
   while1_while2_start = omp_get_wtime(); 		
#endif

	while( edgeMin != NULL )
	{
		updateEdgeVec.push_back( edgeMin->e2 );
		edgeMin = edgeMin->next;
	}

#ifdef WATERSHED_TIMING_CPU
	while1_while2_end = omp_get_wtime();
	sum_time_while1_while2 +=
	    (double)(while1_while2_end - 
		while1_while2_start);
#endif


#ifdef WATERSHED_TIMING_CPU
	while1_phase1_end = omp_get_wtime();
	sum_time_while1_phase1 +=
	       (double)(while1_phase1_end -
		while1_phase1_start);	
#endif

#ifdef WATERSHED_TIMING_CPU
	while1_phase2_start = omp_get_wtime();
#endif
		
#ifdef WATERSHED_TIMING_CPU
	while1_sort_start = omp_get_wtime();
#endif
	//remove duplicates
	sort(updateEdgeVec.begin(), updateEdgeVec.end());

#ifdef WATERSHED_TIMING_CPU
	while1_sort_end = omp_get_wtime();
	sum_time_while1_sort +=
   	    (double)(while1_sort_end -
	    while1_sort_start);
#endif
	
#ifdef WATERSHED_TIMING_CPU
	while1_phase2_p1_start = omp_get_wtime();
#endif
		vector<unsigned int>::iterator it = unique(updateEdgeVec.begin(), updateEdgeVec.end());
		updateEdgeVec.resize( distance(updateEdgeVec.begin(),it) );//unique does not resize vector

 
#ifdef WATERSHED_TIMING_CPU
	while1_phase2_p1_end = omp_get_wtime();
	sum_time_while1_phase2_p1 +=
		(double)(while1_phase2_p1_end - 
		while1_phase2_p1_start);
#endif
		//cout<<"Check point 7"<<endl; //TODO: this is teh functions that seg faults for large specific datasets
		//insert the new node (new region) into node e1 and delete e2. We have to recalculate all the edges from auxNode to all the edges between e1 and e2
		
#ifdef WATERSHED_TIMING_CPU
	while1_phase2_p2_start = omp_get_wtime();
#endif
		agglomerateGraph.mergeNodes(e1, e2, auxNode ,mergeNodesFunctorPtr);


#ifdef WATERSHED_TIMING_CPU
	while1_phase2_p2_end = omp_get_wtime();
	sum_time_while1_phase2_p2 +=
		(double)(while1_phase2_p2_end - 
		while1_phase2_p2_start);
#endif

#ifdef WATERSHED_TIMING_CPU
	while1_for2_start = omp_get_wtime();
#endif
		//cout<<"Check point 8"<<endl;
		//update min tau for affected edges
		for(vector<unsigned int>::iterator iter = updateEdgeVec.begin(); iter != updateEdgeVec.end(); ++iter)
		{
			minTauVec[*iter].first = findEdgeWithMinTau(*iter, agglomerateGraph, minTauVec[ *iter].second );
		}


#ifdef WATERSHED_TIMING_CPU
	while1_for2_end = omp_get_wtime();
	sum_time_while1_for2 +=
	    (double)(while1_for2_end - 
	    while1_for2_start);
#endif

#ifdef WATERSHED_TIMING_CPU
	while1_phase2_p3_start = omp_get_wtime();
#endif
    //check that agglomeration was correct 
    if( numNodes != agglomerateGraph.getNumNodes_Omp() + 1)
    {
      cout<<"Check point 2: iter="<<(iterD)<<";numNodes = "<<numNodes<<".Selected nodes:e1="<<e1<<";e2="<<e2<<";tauE="<<tauE<<endl;

      cout<<"ERROR: buildHierarchicalSegmentation: numNodes = "<<numNodes<<"; numNodes (after agglomeration ) = "<<agglomerateGraph.getNumNodes_Omp()<<endl;
 
      exit(3);
    }

#ifdef WATERSHED_TIMING_CPU
	while1_phase2_p3_end = omp_get_wtime();
	sum_time_while1_phase2_p3 +=
		(double)(while1_phase2_p3_end - 
		while1_phase2_p3_start);
#endif
		numNodes--;

		iterD++;


#ifdef WATERSHED_TIMING_CPU
	while1_phase2_end = omp_get_wtime();
	sum_time_while1_phase2 +=
		(double)(while1_phase2_end - 
		 while1_phase2_start);
#endif
	}

	
cout<<"in buildH while1 loop " << count_while1 <<
	" times"<<endl;
#ifdef WATERSHED_TIMING_CPU
	cout << "in buildH while1 for1 took "
		<<sum_time_while1_for1<<
		" secs"<<endl;

	cout << "in buildH while1 for2 took "
		<<sum_time_while1_for2<<
		" secs"<<endl;
#endif


#ifdef WATERSHED_TIMING_CPU
	cout << "int buildH while1 while1 took "
	   <<sum_time_while1_while1<<
	   " secs"<<endl;
#endif

#ifdef WATERSHED_TIMING_CPU
	cout << "int buildH while1 while2 took "
	   <<sum_time_while1_while2<<
	   " secs"<<endl;
#endif

#ifdef WATERSHED_TIMING_CPU
	cout << "in buildH while1 sort took "
	   <<sum_time_while1_sort<<
	   " secs"<<endl;
#endif

#ifdef WATERSHED_TIMING_CPU
	cout << "in buildH while1 phase1 took "
		<<sum_time_while1_phase1<<
		" secs"<<endl;

	cout << "in buildH while1 phase2 took "
		<<sum_time_while1_phase2<<
		" secs"<<endl;
	cout << "in buildH while1 phase2 p1 took "
		<<sum_time_while1_phase2_p1<<
		" secs"<<endl;
	cout << "in buildH while1 phase2 p2 took "
		<<sum_time_while1_phase2_p2<<
		" secs"<<endl;
	cout << "in buildH while1 phase2 p3 took "
		<<sum_time_while1_phase2_p3<<
		" secs"<<endl;

#endif

	//while1_end = omp_get_wtime();
	//sum_time_while1 +=
	//	(double)(while1_end - 
	//	while1_start);

	//cout << "(new!) in buildH while1 took "
	//	<<sum_time_while1<<
	//	" secs"<<endl;
#ifdef WATERSHED_TIMING_CPU
	while1_end = omp_get_wtime();
	sum_time_while1 +=
		(double)(while1_end - 
		while1_start);

	cout << "in buildH while1 took "
		<<sum_time_while1<<
		" secs"<<endl;
#endif


#ifdef WATERSHED_TIMING_CPU
	time_t while2_start, while2_end;
#endif

#ifdef WATERSHED_TIMING_CPU
	time(&while2_start);
#endif
	//cout<<"DEBUGGING: join disconnected regions"<<endl;
	while( agglomerateGraph.getNumNodes_Omp() > 1 )
	{
		//find two active nodes
		e1 = e2 = std::numeric_limits<unsigned int>::max();
		for(size_t ii = 0; ii<agglomerateGraph.activeNodes.size(); ii++)
		{
			if( agglomerateGraph.activeNodes[ii] == true )
			{
				if( e1 == std::numeric_limits<unsigned int>::max() )
					e1 = ii;
				else if( e2 == std::numeric_limits<unsigned int>::max() )
					e2 = ii;
				else
					break;
			}
		}

		e1NodeOld = agglomerateGraph.nodes[ e1 ];
		
		//create new node in the binary tree 
		auxNode.fMax = max(e1NodeOld.fMax, agglomerateGraph.nodes[ e2 ].fMax);
		auxNode.nodePtr = new TreeNode<nodeHierarchicalSegmentation>();
		auxNode.nodePtr->left = e1NodeOld.nodePtr;
		auxNode.nodePtr->right = agglomerateGraph.nodes[ e2 ].nodePtr;
		auxNode.nodePtr->parent = NULL;
		auxNode.nodePtr->data.svPtr = NULL;//it is not a basic region anymore
		auxNode.nodePtr->data.thrTau = std::numeric_limits<imgVoxelType>::max();

		//update parents for e1 and e2
		e1NodeOld.nodePtr->parent = auxNode.nodePtr;
		agglomerateGraph.nodes[ e2 ].nodePtr->parent = auxNode.nodePtr;
		
		//insert the new node (new region) into node e1 and delete e2. We have to recalculate all the edges from auxNode to all the edges between e1 and e2
		agglomerateGraph.mergeNodes(e1, e2, auxNode ,mergeNodesFunctorPtr);
	}

#ifdef WATERSHED_TIMING_CPU
	time(&while2_end);
	cout << "in buildH while2 took "<<
		difftime(while2_end, while2_start)<<
		" secs"<<endl;
#endif

	//set the root of the binary tree
	hs->dendrogram.SetMainRoot( auxNode.nodePtr );
	


#ifdef WATERSHED_TIMING_CPU
	time(&end);
	cout<<"Building dendrogram took "<<difftime(end,start)<< " secs"<<endl;
#endif

	//--------------------------end of second pass------------------------------------------

	/* Free allocated matrices */
	set_union_destroy(L);
	free(L);
	//PDvec.clear();
	PDhash.clear();
	delete[] neighOffset;   
	free(neighLabel);
	free(neighVal);
	free(neighFmax);


#ifdef USE_MULTITHREAD
	free(neighLabelMT);
	free(neighValMT);
	free(neighFmaxMT);
	delete[] threadData;
#endif

#ifdef WATERSHED_TIMING_CPU
	time(&buildH_end);
	cout<< "in buildH total took "<< 
	 difftime(buildH_end,buildH_start) <<" secs"<< endl;
#endif
	return hs;
}


//==================================================================
GraphEdge* findEdgeWithMinTau(graphFA<nodeAgglomeration>& agglomerateGraph, imgVoxelType& tauE)
{
	imgVoxelType tauAux;
	tauE = imgVoxelMaxValue;
	GraphEdge* auxEdge, *edgeMin = NULL;
	unsigned int e1 = 0;
	for(vector<GraphEdge*>::iterator iterE = agglomerateGraph.edges.begin(); iterE != agglomerateGraph.edges.end(); ++iterE, ++e1)
	{
		auxEdge = (*iterE);
		while( auxEdge != NULL )
		{
			tauAux = min( agglomerateGraph.nodes[e1].fMax, agglomerateGraph.nodes[auxEdge->e2].fMax ) - auxEdge->weight;
			if( tauAux  < tauE)
			{
				tauE = tauAux;
				edgeMin = auxEdge;
			}

			auxEdge = auxEdge->next;
		}		
	}

	return edgeMin;
}

//==================================================================
GraphEdge* findEdgeWithMinTau(unsigned int e1, graphFA<nodeAgglomeration>& agglomerateGraph, imgVoxelType& tauE)
{
	imgVoxelType tauAux;
	tauE = imgVoxelMaxValue;
	GraphEdge* edgeMin = NULL;

	if( e1 >= agglomerateGraph.edges.size() )
		return edgeMin;

	GraphEdge* auxEdge = agglomerateGraph.edges[e1];
	while( auxEdge != NULL )
	{
		tauAux = min( agglomerateGraph.nodes[e1].fMax, agglomerateGraph.nodes[auxEdge->e2].fMax ) - auxEdge->weight;
		if( tauAux  < tauE)
		{
			tauE = tauAux;
			edgeMin = auxEdge;
		}

		auxEdge = auxEdge->next;
	}			
	return edgeMin;
}


