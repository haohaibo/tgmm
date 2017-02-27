/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  lineageHyperTree.h
 *
 *  Created on: August 17th, 2012
 *      Author: Fernando Amat
 *
 * \brief Holds together the hierarchical tree formed by supervoxels, nuclei and lineages. It stores lists with all the nodes in the hypertree to make sure addition and deletions are propagated properly
 *
 */


#ifndef __LINEAGE_HYPERTREE_H__
#define __LINEAGE_HYPERTREE_H__

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
	#include <Windows.h>
#endif

#include <iostream>
#include <vector>
#include <list>
#include <string>
#include "../constants.h"
#include "supervoxel.h"
#include "nuclei.h"
#include "lineage.h"
#include "GaussianMixtureModel_Redux.h"
#include "../nucleiChSvWshedPBC/hierarchicalSegmentation.h"

namespace mylib
{
	#include "array.h"
};

using namespace std;

typedef TreeNode< list<nucleus>::iterator >* rootSublineage;//pointers to original treenode in the lineage

//to hold N dimensional points with very simple operations
template <class T>
struct pointND
{
	T p[dimsImage];

	pointND()
	{
		memset(p,0,dimsImage*sizeof(T));
	};
	pointND(const pointND& other)//copy constructor
	{
		memcpy(p,other.p,sizeof(T)*dimsImage);
	};

	 pointND operator+(const pointND& other)
	 {
		 pointND aux;
		 for(int ii=0;ii<dimsImage;ii++)
			 aux.p[ii] = p[ii] + other.p[ii];
		 return aux;
	 };

	 pointND operator-(const pointND& other)
	 {
		 pointND aux;
		 for(int ii=0;ii<dimsImage;ii++)
			 aux.p[ii] = p[ii] - other.p[ii];
		 return aux;
	 };

	 pointND operator*(const pointND& other)//pointwise multiplication
	 {
		 pointND aux;
		 for(int ii=0;ii<dimsImage;ii++)
			 aux.p[ii] = p[ii] * other.p[ii];
		 return aux;
	 };

	 pointND operator/(const pointND& other)
	 {
		 pointND aux;
		 for(int ii=0;ii<dimsImage;ii++)
			 aux.p[ii] = p[ii] / other.p[ii];
		 return aux;
	 };
	 pointND operator+=(const pointND& other)
	 {
		 for(int ii=0;ii<dimsImage;ii++)
			 p[ii] += other.p[ii];
		 return *this;
	 };
	 pointND operator-=(const pointND& other)
	 {
		 for(int ii=0;ii<dimsImage;ii++)
			 p[ii] -= other.p[ii];
		 return *this;
	 };
	 pointND& operator=(const pointND& other)
	 {
		 if (this != &other)
		 {
			 memcpy(p,other.p,sizeof(T)*dimsImage);
		 }
		 return *this;
	 }
	 pointND operator*=(const T& other)
	 {
		 for(int ii=0;ii<dimsImage;ii++)
			 p[ii] *= other;
		 return *this;
	 };
	 pointND operator/=(const T& other)
	 {
		 for(int ii=0;ii<dimsImage;ii++)
			 p[ii] /= other;
		 return *this;
	 };
	 pointND operator*(const T& other)//scalar multiplication
	 {
		 pointND aux;
		 for(int ii=0;ii<dimsImage;ii++)
			 aux.p[ii] = p[ii] * other;
		 return aux;
	 };
	 pointND operator/(const T& other)//scalar division
	 {
		 pointND aux;
		 for(int ii=0;ii<dimsImage;ii++)
			 aux.p[ii] = p[ii] / other;
		 return aux;
	 };
};

//========================================================================================
class lineageHyperTree
{
public:

	list< supervoxel > *supervoxelsList;//supervoxelsList[ii] contains a list of all the supervoxels at time ii. I need to preallocate memory in order to avoid iterators to list to be invalidated
	list< nucleus > *nucleiList;//supervoxelsList[ii] contains a list of all the nuclei at time ii
	list< lineage > lineagesList;//contains all the lineages

	vector< TreeNode< ChildrenTypeLineage >* > vecRefLastTreeNode;//in case we need to keep a temporary (i.e. short lived) references to a set of nuclei in the lineages (for example, to update TGMM sequential results)

	//list< TreeNode<nodeHierarchicalSegmentation>* > *supervoxelsHSnodeList;//supervoxelsList[ii][jj] contains a pointer to the node in the hierarchical segmentation that generates this supervoxel (it could be null );

	//constructor/destructor functions
	void clear();//cleans hypetree but does not deallocate memory
	lineageHyperTree();
	lineageHyperTree(unsigned int maxTM_);
	lineageHyperTree(const lineageHyperTree & p);
	~lineageHyperTree();
	void releaseData();
	ParentTypeSupervoxel addNucleusFromSupervoxel(unsigned int TM, ChildrenTypeNucleus& sv);//creates a nucleus from a single supervoxel and updates the hypergraph properly. Returns an iterator to the created element
	void deleteLineage(list< lineage >::iterator& iterL);//deletes all the nuclei associated with a lineage and the lineage ityself. iterL is updated with the next element in lineagesList

	//simple set/get functions
	void getSupervoxelListIteratorsAtTM(vector< list<supervoxel>::iterator > &vec, unsigned int TM);
	void getNucleiListIteratorsAtTM(vector< list<nucleus>::iterator > &vec, unsigned int TM);
	void getLineageListIterators(vector< list<lineage>::iterator > &vec);
	void setIsSublineage(bool p){isSublineage = p;};
	void setRootSublineage(vector< rootSublineage > &p){ sublineageRootVec = p;};
	int getMaxTM(){return maxTM;};
	bool getIsSublineage(){return isSublineage;};
	int findRootSublineage(TreeNode< ChildrenTypeLineage >* p);//returns -1 if root is not found
	void setFrameAsT_o(unsigned int TM);//deletes anything < TM. All the nuclei existing at TM are set as root of the starting lineages


	//disconnet / connect elements between hypergraphs
	void disconnectNucleusFromSupervoxel(nucleus &nuc, bool killSupervoxels);

	//tree traversing operation
	TreeNode<ChildrenTypeLineage>* findRoot(TreeNode<ChildrenTypeLineage>* root, int minTM = -1);//goes upstream the lineage up to minTM to return parent of TreeNode root

	//I/O functions
	int readListSupervoxelsFromTif(string filename,int vecPosTM);
	int readListSupervoxelsFromTifWithWeightedCentroid(string filenameImage,string filenameLabels,int vecPosTM);
	int readBinaryRnkTGMMfile(string filenameRnk,int vecPosTM);//reads Rnk binary file to associate supervoxels into nuclei
	int parseTGMMframeResult(vector<GaussianMixtureModelRedux*> &vecGM, int vecPosTM);//uses vecRefLastTreeNode to keep track of where each new nuclei needs to be connected to 
	void parseImagePath(string& imgRawPath, int frame);//given a filename with ???? decodes it to find the file
	int parseNucleiList2TGMM(vector<GaussianMixtureModelRedux*> &vecGM, int vecPosTM);//parse information in nuclei list to a GMM data structure
	int writeListSupervoxelsToBinaryFile(string filename, unsigned int TM);
	int writeLineageInArray(string filename);//saves lineage (only nuclei centroids) as an array with columns id, type, x, y, z, radius, parent_id, time, confidence
	int readLineageInArray(string filename);

	//operators
	lineageHyperTree& operator=(const lineageHyperTree & p);

	//extraction of subgraphs to resolve death and cell division issues
	//WARNING: each root must belong to a separate lineage!!!
	int cutSublineage (vector< rootSublineage > &vecRoot, lineageHyperTree& lht);//extracts a sublineage starting at vecRoot[ii] for lengthTM time points. lht will contain the sublineage
	int cutSingleSublineageFromRoot(rootSublineage& root); //extracts sublineage for a single root and ADS it. So you can build
	int pasteOpenEndSublineage (lineageHyperTree& lhtOrig);//paste back a sublineage that goes all teh way to the end of the lineage, so we do not need to worry about how to paste children
	int cutSublineageCellDeathDivisionEvents(unsigned int winRadiusTime,unsigned int TM, vector< lineageHyperTree >& lhtVec);//we cut sublineages with window \in [TM-winRadiusTime,TM+winRadiusTime]
	

	int supervoxelNearestNeighborsInSpace(unsigned int TM,unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA = 0);//you need to setup scale to calculate this properly
	int supervoxelNearestNeighborsInSpace(unsigned int TM, int devCUDA = 0){return supervoxelNearestNeighborsInSpace(TM,supervoxel::getKmaxNumNN(), supervoxel::getKmaxDistKNN(), devCUDA);};
	int supervoxelNearestNeighborsInTimeForward(unsigned int TM,unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA = 0);//you need to setup scale to calculate this properly
	int supervoxelNearestNeighborsInTimeForward(unsigned int TM, int devCUDA = 0){return supervoxelNearestNeighborsInTimeForward(TM,supervoxel::getKmaxNumNN(), supervoxel::getKmaxDistKNN(), devCUDA);};
	int supervoxelNearestNeighborsInTimeBackward(unsigned int TM,unsigned int KmaxNumNN, float KmaxDistKNN, int devCUDA = 0);//you need to setup scale to calculate this properly
	int supervoxelNearestNeighborsInTimeBackward(unsigned int TM, int devCUDA = 0){return supervoxelNearestNeighborsInTimeBackward(TM,supervoxel::getKmaxNumNN(), supervoxel::getKmaxDistKNN(), devCUDA);};
	static float findNearestNucleusNeighborInSpaceEuclideanL2(const ChildrenTypeLineage& iterNucleus, ChildrenTypeLineage& iterNucleusNN) ;//uses supervoxels to find nearest neighbor. Returns 1e32 if nothing was found
	static int findKNearestNucleiNeighborInSpaceEuclideanL2(const ChildrenTypeLineage& iterNucleus, vector<ChildrenTypeLineage>& iterNucleusNNvec, vector<float>& distVec);//uses supervoxels to find K nearest neighbors. Returns 1e32 if nothing was found	
	static int findKNearestNucleiNeighborInTimeForwardEuclideanL2(const ChildrenTypeLineage& iterNucleus, vector<ChildrenTypeLineage>& iterNucleusNNvec, vector<float>& distVec);//uses supervoxels to find K nearest neighbors. Returns 1e32 if nothing was found
	
	static int findKNearestNucleiNeighborInTimeForwardSupervoxelEuclideanL2(const ChildrenTypeLineage& iterNucleus, vector<ChildrenTypeLineage>& iterNucleusNNvec, vector<float>& distVec);//uses supervoxels to find K nearest neighbors. Returns 1e32 if nothing was found. IT IS NOT THE DISTANCE BETWEEN NUCLEI BUT BETWEEN SUPERVOXELS
	static int findKNearestNucleiNeighborInSpaceSupervoxelEuclideanL2(const ChildrenTypeLineage& iterNucleus, vector<ChildrenTypeLineage>& iterNucleusNNvec, vector<float>& distVec);//uses supervoxels to find K nearest neighbors. Returns 1e32 if nothing was found. IT IS NOT THE DISTANCE BETWEEN NUCLEI BUT BETWEEN SUPERVOXELS

	//statistics
	float deathJaccardRatio(TreeNode<ChildrenTypeLineage>* rootSplit) const;//checks Jaccard distance ratios to see if one cell has invaded teh space of another after death
	void deathJaccardRatioAll(vector<float>& ll, TreeNode<ChildrenTypeLineage>* mainRoot) const;
	int daughterLengthToNearestNeighborDivision(TreeNode<ChildrenTypeLineage>* rootSplit) const;//length in time points before the nearest neighbor of a dead cell splits. If it dies, the length to death is returned as a negative integer
	void daughterLengthToNearestNeighborDivisionAll(vector<int>& ll, TreeNode<ChildrenTypeLineage>* mainRoot) const;
	template<class imgTypeC>
	int calculateNucleiIntensityCentroid(SibilingTypeNucleus& iterN);//recalculates centroid based on intensity. If data is not present then it is just a mean of coordinate

	template<class imgTypeC>
	void calculateGMMparametersFromNuclei(list<nucleus>::iterator &iterN, float* m_k, float *N_k, float* S_k);//calculates statistics for the Gaussian based on supervoxls belonging to centroid. We assume r_nk =1

	//metrics
	static float offsetJaccardDistance(TreeNode<ChildrenTypeLineage>* node); //Jaccard distance between two nuclei after translation using centroids

	/*
	\param[in] thrDist2: threshold distance (squared Euclidean with scale) between nucleus and its parent to be considered a suspicious displacement
	*/
	int confidenceScoreForNucleus(TreeNode<ChildrenTypeLineage>* rootConfidence, float thrDist2);//calculates a score between 0 [not confident] and 5 [very confident] to be used by CATMAID editor in order to speed up fixing of errors

	//simple heuristic rules to correct lineages
	int mergeShortLivedDaughters(TreeNode<ChildrenTypeLineage>* rootSplit, int lengthTMthr);//for each cell division, checks when any of the the daughters die. If it is below lengthTMthr, then it undoes the split and merges both daughters 
	int mergeShortLivedDaughtersAll(int lengthTMthr, int maxTM, int& numCorrections, int& numSplits);// checks for all possible corrections in *this. maxTM in order to avoid conisdering dead the "forefront" of tracking. Just set maxTM = maxINT to ignore this constraint
	int deleteShortLivedDaughters(TreeNode<ChildrenTypeLineage>* rootSplit, int lengthTMthr);//for each cell division, checks when any of the the daughters die. If it is below lengthTMthr, deletes that part of the lineage
	int deleteShortLivedDaughtersAll(int lengthTMthr, int maxTM, int& numCorrections, int& numSplits);// checks for all possible corrections in *this. maxTM in order to avoid conisdering dead the "forefront" of tracking. Just set maxTM = maxINT to ignore this constraint
	int splitDeathDivisionPattern(TreeNode<ChildrenTypeLineage>* rootSplit, int lengthTMthr);//corrects the pattern of a dead cell wher eits neighbor splits right after (taking over the spot)
	int mergeNonSeparatingDaughters(TreeNode<ChildrenTypeLineage>* rootSplit,int conn3D, int64* neighOffset);//for each cell division, checks the distance over time between the two daughters. If they are always neighboring, it merges them.
	int mergeNonSeparatingDaughters(TreeNode<ChildrenTypeLineage>* rootSplit,int conn3D, int64* neighOffset, size_t minNeighboringVoxels);
	int mergeNonSeparatingDaughtersAll(int maxTM, size_t minNeighboringVoxels, int conn3D, int& numCorrections, int& numSplits);
	int mergeShortLivedAndCloseByDaughtersAll(int lengthTMthr,int maxTM, size_t minNeighboringVoxels, int conn3D, int& numCorrections, int& numSplits);// apply both mergeShortLivedDaughters and mergeNonSeparatingDaughters
	int mergeParallelLineages(TreeNode<ChildrenTypeLineage>* root1, TreeNode<ChildrenTypeLineage>* root2,int conn3D, int64* neighOffset, size_t minNeighboringVoxels);//very similar to mergeNonSeparatingDaughters but with neighboring lineages
	int mergeParallelLineagesAll(int conn3D, size_t minNeighboringVoxels, int &numMerges);
	int extendDeadNuclei(TreeNode<ChildrenTypeLineage>* rootDead);//for a dead cell, tries to find an extension. Usually some other cell has "overtaken" the space, so this function creates a feasible extension to continue both tracks 
	int extendDeadNucleiAtTM(int TM, int& numExtensions, int &numDeaths);// checks for all possible corrections in *this. 
	int deleteShortLineagesWithHighBackgroundClassifierScore(int maxLengthBackgroundCheck, int frame, float thrMinScoreFx, int& numPositiveChecks, int numActions);//finds lineages shorter than maxLengthBackgroundCheck and if all the supervoxels qualify as background->delete

	int breakCellDivisionBasedOnCellDivisionPlaneConstraint(int TM, double thrCellDivisionPlaneDistance ,int &numCorrections, int& numSplits);

	int deleteDeadBranchesAll(int maxTM, int &numDelete);//delete lineages that die before even dividing up to maxTM point
	



	/*
	\brief find how many clusters of supervoxels we have based on conn3D and minNeighboringVoxels "touching" criteria

	\param[out] supervoxelIdx For each supervoxel belonging to rootNuclei returns the id of the clustering they belong to

	\returns number of clusters. Negative number if error.
	*/
	int numberOfSeparatedSupervoxelClustersInNucleus(TreeNode<ChildrenTypeLineage>* rootNuclei,int conn3D, int64* neighOffset, size_t minNeighboringVoxels, vector<int>& supervoxelIdx);

	//functions to modify edges in the hierarchical graph
	//VIP: WE ALWAYS MERGE ROOT2 TO ROOT1 (SO ROOT2 IS DESTROYED). THIS IS IMPORTANT IF THEY BELONG TO DIFFERENT LINEAGES
	int mergeBranches(TreeNode<ChildrenTypeLineage>* root1 ,TreeNode<ChildrenTypeLineage>* root2);//merges two branches (all the way until one dies) starting at rootX (root1 and root2 are also merged)

	int deleteBranch(TreeNode<ChildrenTypeLineage>* root);//it disconnects root1 from parent and deletes all the sublineage taking root as root node

	//debugging functions to check results
	int debugCheckHierachicalTreeConsistency();//checks that child(p.parent) = p
	void debugPrintLineage(int lineageNumber);
	void debugPrintNucleus(int TM, int nucleusNumber);
	void debugPrintLineageForLocalLineageDisplayinMatlab(string imgPath, string imgLPath, string suffix = "", string imgRawPath = "");
	void debugPrintLineageToMCMCxmlFile(string filename,string imgPrefix, string imgSuffix);
	void debugCanvasFromSegmentationCroppedRegion(unsigned int TM, uint64 minXYZ[dimsImage], uint64 maxXYZ[dimsImage],string imgBasename, string imgRawPath ="");
	void debugPrintGMMwithDisconnectedSupervoxels(unsigned int TM, int minNumDisconnectedSv, int conn3D, size_t minNeighboringVoxels, vector<int>& nucleiIdx);
	void alphaBlend(mylib::Array* imgRGB, mylib::Array* imgLabels, double alpha);//alpha blend functions based on mylib::Draw_Partition but without having to make partitions
	uint64 debugNumberOfTotalNuclei();
	int debugCheckOneToOneCorrespondenceNucleiSv(int TM);

protected:

private:

	const unsigned int maxTM;//needs to be declared at the beginning and set to an upper bound of the number of time points
	vector< rootSublineage > sublineageRootVec; //contains the pointers to each starting nuclei to extract a sublineage. There can be multiple roots. Only used if isSublineage = true.
	bool isSublineage; //to know if it is a subgrpah or not. If hyperTree is a sublineage, maxTM indicates its length

	//functions used by o ther methods but that users should not need	 
	int pasteSingleOpenEndSublineageFromRoot(lineageHyperTree& lhtOrig, size_t pos);
	TreeNode< ChildrenTypeLineage >* CopyPartialLineage(TreeNode< ChildrenTypeLineage > *root, TreeNode< ChildrenTypeLineage > *parent, int boundsTM);//modified from binaryTree.h to just copy a portion of a lineage
	int cutSublineageCellDeathDivisionEventsRecursive(int minTM, vector< lineageHyperTree >& lhtVec , queue< TreeNode<ChildrenTypeLineage>* >& q);//used by cutSublineageCellDeathDivisionEvents to build sublineage recursively
};




//===============================================================================
inline int lineageHyperTree::findRootSublineage(TreeNode< ChildrenTypeLineage >* p)
{
	int count = 0;
	for(vector<TreeNode< ChildrenTypeLineage >* >::const_iterator iter = sublineageRootVec.begin(); iter != sublineageRootVec.end(); ++iter)
	{
		if( (*iter) == p)
			return count;
		count++;
	}
	return -1;
}

inline void lineageHyperTree::releaseData()
{
	for(unsigned int ii =0;ii<maxTM; ii++)
	{
		if( supervoxelsList[ii].front().dataPtr != NULL )//we assume all supervoxels within the same time point are pointing to teh same dataset
		{
			free(supervoxelsList[ii].front().dataPtr);//we assume it was created with malloc (most likely with mylib)
			for(list<supervoxel>::iterator iter = supervoxelsList[ii].begin(); iter != supervoxelsList[ii].end(); ++iter)
				iter->dataPtr = NULL;
		}
	}
}


//====================================================
inline void lineageHyperTree::getSupervoxelListIteratorsAtTM(vector< list<supervoxel>::iterator > &vec, unsigned int TM)
{
	

	if(TM >= maxTM)
	{
		cout<<"WARNING at: lineageHyperTree::getSupervoxelListIteratorsAtTM: time point requested does not exist.  You need to make maxTM larger and recompile the code."<<endl;
		return;
	}

	vec.resize(supervoxelsList[TM].size());
	vector< list<supervoxel>::iterator >::iterator vecIter = vec.begin();
	for(list<supervoxel>::iterator iter = supervoxelsList[TM].begin(); iter != supervoxelsList[TM].end(); ++iter,++vecIter)
	{
		(*vecIter) = iter;
	}

	return;
}

inline void lineageHyperTree::getNucleiListIteratorsAtTM(vector< list<nucleus>::iterator > &vec, unsigned int TM)
{
	

	if(TM >= maxTM)
	{
		vec.clear();
		cout<<"WARNING at: lineageHyperTree::getNucleiListIteratorsAtTM: time point requested does not exist. You need to make maxTM larger and recompile the code."<<endl;
		return;
	}

	vec.resize(nucleiList[TM].size());
	vector< list<nucleus>::iterator >::iterator vecIter = vec.begin();
	for(list<nucleus>::iterator iter = nucleiList[TM].begin(); iter != nucleiList[TM].end(); ++iter,++vecIter)
	{
		(*vecIter) = iter;
	}

	return;
}

inline void lineageHyperTree::getLineageListIterators(vector< list<lineage>::iterator > &vec)
{
	vec.resize(lineagesList.size());
	vector< list<lineage>::iterator >::iterator vecIter = vec.begin();
	for(list<lineage>::iterator iter = lineagesList.begin(); iter != lineagesList.end(); ++iter,++vecIter)
	{
		(*vecIter) = iter;
	}

	return;
}

//===============================================
inline void  lineageHyperTree::disconnectNucleusFromSupervoxel(nucleus &nuc, bool killSupervoxels)
{
	if ( killSupervoxels == true )
	{
		for(vector< ChildrenTypeNucleus >::iterator iterS = nuc.treeNode.getChildren().begin(); iterS != nuc.treeNode.getChildren().end(); ++iterS )
		{
			(*iterS)->treeNode.deleteParent();
			//kill supervoxels: we do not erase them from the list (to avoid problems with pointers), but we remove its main features
			(*iterS)->PixelIdxList.clear();
			(*iterS)->PixelIdxList.push_back( 0 );//to avoid crashing other parts of the code with empty supervoxels
			(*iterS)->centroid[0] = -1e32;
		}
	}else{//no need to kill supervoxels
		for(vector< ChildrenTypeNucleus >::iterator iterS = nuc.treeNode.getChildren().begin(); iterS != nuc.treeNode.getChildren().end(); ++iterS )
		{
			(*iterS)->treeNode.deleteParent();
		}
	}
	nuc.treeNode.deleteChildrenAll();
}



#endif
