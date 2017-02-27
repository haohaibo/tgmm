/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  hierachicalSegmentation.h
 *
 *  Created on: January 17th, 2013
 *      Author: Fernando Amat
 *
 * \brief Data structure to save a hierarchical segmentation of an image
 *
 */


#ifndef __HIERARCHICAL_SEGMENTATION_H__
#define __HIERARCHICAL_SEGMENTATION_H__

#include <fstream>
#include "constants.h"
#include "supervoxel.h"
#include "binaryTree.h"


//simple structure to hold a node in the hierarchical segmentation
struct nodeHierarchicalSegmentation
{
	imgVoxelType thrTau;//tau value for which the two children of this node merge. The thrTau of any descendant of this node should be lower
	supervoxel* svPtr;//if the node is a leaf node (basic region), it points to basicRegionsVec that represents this leave node. Otherwise it should be NULL;

	nodeHierarchicalSegmentation()
	{
		svPtr = NULL;
	}

	~nodeHierarchicalSegmentation()
	{
		//nothing to do: this function does not deallocate svPtr;
	}

	nodeHierarchicalSegmentation(const nodeHierarchicalSegmentation& p)
	{
		thrTau = p.thrTau;
		svPtr = p.svPtr;
	}
	nodeHierarchicalSegmentation& operator=(const nodeHierarchicalSegmentation& p);

	nodeHierarchicalSegmentation(istream& is, supervoxel* basicRegionsVec);//create supervoxel from biunary file
	void writeToBinary(ostream& os);//write ot binary file
};



//==================================================================================
class hierarchicalSegmentation
{

public:
	supervoxel* basicRegionsVec;//vector with the supervoxels containing teh basic regions (leaves of the dendrogram)	
	BinaryTree<nodeHierarchicalSegmentation> dendrogram;//stores the complete hierarchical segmentation

	vector< TreeNode<nodeHierarchicalSegmentation>* > currentSegmentationNodes;//contains current segmentation (pointer to nodes in dendrogram). It is a specific "cut" of the dendrogram
	vector< supervoxel >							  currentSegmentatioSupervoxel;//contains current segmentation (supervoxels). It should always be the same size as currentSegmentationNodes


	//constructor destructors
	hierarchicalSegmentation();
	hierarchicalSegmentation(unsigned int numBasicRegions_);
	hierarchicalSegmentation(unsigned int numBasicRegions_, int TM);
	hierarchicalSegmentation(const hierarchicalSegmentation& p);
	~hierarchicalSegmentation();

	hierarchicalSegmentation(istream& is);//create supervoxel from biunary file
	void writeToBinary(ostream& os);//write ot binary file

	//set / get functions
	void setMaxTau(imgVoxelType t){maxTau = t;};
	imgVoxelType getMaxTau() const {return maxTau;};
	int getNumberOfDescendants( TreeNode<nodeHierarchicalSegmentation>*  hsNode);//number of nodes under hsNode in the dendrogram (we include the node itself). So if it is a leave, the code returns 1

	//basic set/get I/O functions
	unsigned int getNumberOfBasicRegions() { return numBasicRegions;};
	unsigned int getNumberOfBasicRegions() const{ return numBasicRegions;};
	int shrinkBasicRegionsVec(unsigned int ss)
	{
		if( ss > numBasicRegions )
		{
			cout<<"ERROR:shrinkBasicRegionsVec: you are trying to increase the size of numBasicRegionsVec"<<endl;
			return 1;
		}else{
			numBasicRegions = ss;
			return 0;
		}
	};
	void resetNodeIdDendrogram(vector<TreeNode<nodeHierarchicalSegmentation>*> &mapNodId2ptr);//so we can recover later from binary file (instead of using pointers). Basicallyy nodes are labeled traversing the graph in a breadth first manner;

	//main functions to operate on dendrogram and create specific segmentations	
	/*
	\brief creates a segmentation for a given tau from the tree of hierarchical segmentations.

	\param[in]	tau								threshold to generate segmentation
	\param[out] currentSegmentationNodes		returns pointers to the nodes in the dendrogram corresponding to the regions. To generate supervoxels from these pointers use the method supervoxelAtTreeNode. This variable is part of teh class
	*/
	int segmentationNodesAtTau(imgVoxelType tau);
	int segmentationAtTau(imgVoxelType tau);//generates both tree nodes and supervoxels

	/*
	\brief Generates a supervoxel from a node in the dendogram. Briefly, it finds all the descendants that are leaves and merges all those basic regions to create a supervoxel
	\warning It does not calculate centroid or any other properties of the supervoxel
	*/
	int supervoxelAtTreeNode(TreeNode<nodeHierarchicalSegmentation>* hsNode, supervoxel& sv);
	size_t supervoxelAtTreeNodeOnlySize(TreeNode<nodeHierarchicalSegmentation>* hsNode);//it does not calculate teh supervoxel, just the size


	/*
	\brief Find all the descendants from root that are active in current segmentation. This can be useful in split/merge situations to impose exclusivity constraints due to hierarchy
	*/
	void findDescendantsInCurrentSegmentation(TreeNode<nodeHierarchicalSegmentation>* root, vector< TreeNode<nodeHierarchicalSegmentation>* >& vecD);

	/*
	\brief Efficiently erases a list of supervoxels from the current segmentation. WARNING: it uses swap and resize, so order of superovxels is altered

	\param[in]  eraseIdx: it has to be SORTED (ASCENDING) UNIQUE set of indexes to erase
	*/
	void eraseSupervoxelFromCurrentSegmentation(const vector<int>& eraseIdx);


	/*
	\brief propose merge in the segmentation from hierarchical clustering

	\warning it uses static variables from supervoxel class in order to trim super voxel

	\param in	root			Node in which we want to propose merge
	\param in	rootSv			supervoxel associated with root
	\param out	rootMerge		Node proposed to merge to root. NULL means there was no possible merge
	\param out	rootMergeSv		supervoxel associated with rootMerge
	\return						Score from [0,1] on probability of being a correct merge
	*/
	template<class imgTypeC>
	float suggestMerge(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>** rootMerge,  supervoxel& rootMergeSv, int debugRecursion = 0);


	/*
	\brief propose split in the segmentation from hierarchical clustering

	\warning it uses static variables from supervoxel class in order to trim super voxel

	\param in	root			Node in which we want to propose split
	\param in	rootSv			supervoxel associated with root
	\param out	rootSplit		Nodes proposed to split root. NULL means there was no possible merge
	\param out	rootSplitSv		supervoxel associated with rootSplit
	\return						Score from [0,1] on probability of being a correct split
	*/
	template<class imgTypeC>
	float suggestSplit(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>* rootSplit[2],  supervoxel rootSplitSv[2]);


	//---------other functions---------------
	void findVeryPersistantBasicRegions( vector<supervoxel*>& svVec);//leaves nodes in which parent node has tau>tauMax -> very stable minima
	void findRealisticNodes( unsigned int minSize);//creates a new "segmentation" with  all the nodes in the binary tree that satisify certain basic conditions

	template<class imgTypeC>
	void cleanHierarchyWithTrimming( unsigned int minSize, unsigned int maxSize, int devCUDA); //calls findRealistic nodes and it also trims supervoxels in order to remove duplicates from hierarchical segmentation

	/*
	\brief Refined version of cleanHierarchyWithTrimming. It uses simple checks to disconnect remove duplicate nodes in the hierarchy

	\warning it uses static variables from supervoxel class in order to trim super voxel
	*/
	template<class imgTypeC>
	void cleanHierarchy(); 

	//-------------------------debug functions--------------------------------------
	int debugCheckDendrogramCoherence();
	int debugTestMergeSplitSuggestions(imgVoxelType tau);
	int debugNumberOfNodesBelowTauMax();
	void debugHierarchyDepth(string filename);
	void debugEstimateDeltaZsupervoxel(imgVoxelType tau, string filename);

protected:


private:
	unsigned int numBasicRegions;
	imgVoxelType maxTau;//indicates the maximum tau allowed to merge two nodes in the dendrogram. It is used in many functions

};





//===================================================================================
inline int hierarchicalSegmentation::getNumberOfDescendants( TreeNode<nodeHierarchicalSegmentation>*  hsNode)//number of nodes under hsNode in the dendrogram
{
	if( hsNode == NULL )
		return 0;

	int n = 0;
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( hsNode );
	TreeNode<nodeHierarchicalSegmentation>* auxNode;


	while(q.empty() == false )
	{
		auxNode = q.front();
		q.pop();
		n++;
		if( auxNode->left != NULL )
			q.push( auxNode->left );
		if( auxNode->right != NULL )
			q.push( auxNode->right );

	}

	return n;
}

#endif
