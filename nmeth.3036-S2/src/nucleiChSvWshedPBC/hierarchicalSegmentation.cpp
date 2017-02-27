/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  hierachicalSegmentation.cpp
 *
 *  Created on: January 17th, 2013
 *      Author: Fernando Amat
 *
 * \brief Data structure to save a hierarchical segmentation of an image
 *
 */


/* for memory checking
#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <Windows.h>
#include <Psapi.h>
#include <stdio.h>
#pragma comment(linker, "/DEFAULTLIB:psapi.lib")
#endif
*/

#include <queue>//FIFO
#include <deque>//double ended queue
#include <string>
#include <math.h>
#include <limits>
#include <algorithm>
#include "hierarchicalSegmentation.h"



nodeHierarchicalSegmentation& nodeHierarchicalSegmentation::operator=(const nodeHierarchicalSegmentation& p)
{
	if (this != &p)
	{
		thrTau = p.thrTau;
		svPtr = p.svPtr;
	}
	return *this;
}

nodeHierarchicalSegmentation::nodeHierarchicalSegmentation(istream& is, supervoxel* basicRegionsVec)//create supervoxel from biunary file
{
	is.read((char*) (&thrTau),sizeof(imgVoxelType));
	int ii;
	is.read((char*) ( &(ii) ),sizeof(int));//to know the order
	if( ii == -1 )
		svPtr = NULL;
	else
		svPtr = basicRegionsVec + ii;	

}
void nodeHierarchicalSegmentation::writeToBinary(ostream& os)//write ot binary file
{
	os.write((char*) (&thrTau),sizeof(imgVoxelType));
	int ii;
	if( svPtr == NULL )
		ii = -1;
	else
		ii = (int) (svPtr->tempWildcard);//wild card needs to store the order

	os.write((char*) ( &(ii) ),sizeof(int));//to know the order
}
//===============================================================================
hierarchicalSegmentation::hierarchicalSegmentation()
{
	numBasicRegions = 0;
	basicRegionsVec = NULL;
	maxTau = std::numeric_limits<imgVoxelType>::max();
};


hierarchicalSegmentation::hierarchicalSegmentation(unsigned int numBasicRegions_)
{
	numBasicRegions = numBasicRegions_;
	basicRegionsVec = new supervoxel[numBasicRegions];
	maxTau = std::numeric_limits<imgVoxelType>::max();
};

hierarchicalSegmentation::hierarchicalSegmentation(unsigned int numBasicRegions_, int TM)
{
	numBasicRegions = numBasicRegions_;
	basicRegionsVec = new supervoxel[numBasicRegions];

	for(unsigned int ii = 0;ii<numBasicRegions; ii++)
		basicRegionsVec[ii].TM = TM;

	maxTau = std::numeric_limits<imgVoxelType>::max();
};

hierarchicalSegmentation::hierarchicalSegmentation(const hierarchicalSegmentation& p)
{
	numBasicRegions = p.getNumberOfBasicRegions();
	if( numBasicRegions > 0)
	{
		basicRegionsVec = new supervoxel[numBasicRegions];
		for(unsigned int ii = 0;ii<numBasicRegions; ii++)
		{
			basicRegionsVec[ii] = p.basicRegionsVec[ii];
		}
	}
	else
		basicRegionsVec = NULL;

	maxTau = p.getMaxTau();
};


hierarchicalSegmentation::~hierarchicalSegmentation()
{
	if( basicRegionsVec != NULL)
	{
		delete[] basicRegionsVec;
		basicRegionsVec = NULL;
	}

	dendrogram.clear();
};

hierarchicalSegmentation::hierarchicalSegmentation(istream& is)//create supervoxel from biunary file
{
	is.read((char*) ( &(numBasicRegions) ),sizeof(unsigned int));

	basicRegionsVec = new supervoxel[numBasicRegions];
	for(unsigned int ii = 0;ii<numBasicRegions;ii++)
	{
		basicRegionsVec[ii] = supervoxel(is);
	}

	//reconstruct tree
	int numNodes;
	is.read((char*) ( &(numNodes) ),sizeof(int));
	
	vector<int> parentIdx(numNodes);
	vector< TreeNode<nodeHierarchicalSegmentation>* > nodes(numNodes);

	for(int ii =0; ii < numNodes; ii++)
	{
		nodes[ii] = new TreeNode<nodeHierarchicalSegmentation>();
		nodes[ii]->data = nodeHierarchicalSegmentation( is, basicRegionsVec);
		is.read((char*) ( &(parentIdx[ii]) ),sizeof(int));
	}

	if( parentIdx[0] != -1 )//root
	{
		cout<<"ERROR: at hierarchicalSegmentation: binary file seems corrupted. First node is not root"<<endl;
		exit(3);
	}
	dendrogram.SetMainRoot( nodes[0] );

	for(int ii = 1; ii < numNodes; ii++)
	{
		if( parentIdx[ii] < 0 )
		{
			cout<<"ERROR: at hierarchicalSegmentation: binary file seems corrupted. Node " <<ii<<" is also root"<<endl;
			exit(3);
		}
		nodes[ii]->parent = nodes[ parentIdx[ii] ];

		if( nodes[ parentIdx[ii] ]->left == NULL )
		{
			nodes[ parentIdx[ii] ]->left = nodes[ii];
		}else if( nodes[ parentIdx[ii] ]->right == NULL )
		{
			nodes[ parentIdx[ii] ]->right = nodes[ii];
		}else{
			cout<<"ERROR: at hierarchicalSegmentation: node cannot have more children. Node " <<ii<<" is also root"<<endl;
			exit(5);
		}
	}

}

void hierarchicalSegmentation::writeToBinary(ostream& os)//write ot binary file
{
	os.write((char*) ( &(numBasicRegions) ),sizeof(unsigned int));

	for(unsigned int ii = 0;ii<numBasicRegions;ii++)
	{
		basicRegionsVec[ii].tempWildcard = ii;
		basicRegionsVec[ii].writeToBinary( os );
	}

	//traverse tree to set id
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( dendrogram.pointer_mainRoot() );

	TreeNode<nodeHierarchicalSegmentation>* aux;
	int count = 0;
	while( !q.empty() )
	{
		aux = q.front();
		q.pop();

		if( aux->left != NULL ) q.push( aux->left );
		if( aux->right != NULL ) q.push( aux->right );

		aux->nodeId = count;
		count++;
	}

	//traverse the tree to save data
	os.write((char*) ( &(count) ),sizeof(int));//number of nodes
	q.push( dendrogram.pointer_mainRoot() );
	while( !q.empty() )
	{
		aux = q.front();
		q.pop();

		if( aux->left != NULL ) q.push( aux->left );
		if( aux->right != NULL ) q.push( aux->right );

		aux->data.writeToBinary(os);
		if( aux->parent != NULL)
			os.write((char*) ( &(aux->parent->nodeId) ),sizeof(int));
		else{
			int ll = -1;
			os.write((char*) ( &(ll) ),sizeof(int));
		}
	}

}
//===============================================================================

int hierarchicalSegmentation::segmentationNodesAtTau(imgVoxelType tau)
{
	currentSegmentationNodes.clear();

	//explore the tree and keep all the nodes that are below tau for teh first time
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( dendrogram.pointer_mainRoot() );
	TreeNode<nodeHierarchicalSegmentation>* auxNode;

	while ( q.empty() == false )
	{
		auxNode = q.front();
		q.pop();

		if( auxNode->getNumChildren() == 0 )
		{
			//leaf node
			currentSegmentationNodes.push_back( auxNode );
		}else{
			//it should have two children always (it is a dendrogram)
			if( auxNode->data.thrTau <= tau )//we select this region and we do not look at descendants
			{
				currentSegmentationNodes.push_back( auxNode );
			}else{//we keep traversing this branch
				q.push( auxNode->left );
				q.push( auxNode->right );
			}
		}
	}

	return 0;
}

//===============================================================================

int hierarchicalSegmentation::segmentationAtTau(imgVoxelType tau)
{
	//generate nodes
	int err = segmentationNodesAtTau(tau);
	if( err > 0 )
		return err;


	/*
	//reserve allways a little bit more space in case we add some splits afterwards (we do not want to dynamically reallocate too much if we are keeping pointers)
	float extra = (float)(currentSegmentationNodes.size()) * 0.61803;//golden ratio
	extra = std::min( extra, 100.0f );
	extra = std::max ( extra, 10000.0f);
	currentSegmentatioSupervoxel.reserve( currentSegmentationNodes.size() + (size_t)(extra) );
	*/

	//generate supervoxels from nodes
	currentSegmentatioSupervoxel.resize( currentSegmentationNodes.size() );
	

	int count = 0;
	for(vector< TreeNode<nodeHierarchicalSegmentation>* >::iterator iter = currentSegmentationNodes.begin(); iter != currentSegmentationNodes.end(); ++iter, ++count)
	{
		supervoxelAtTreeNode((*iter),currentSegmentatioSupervoxel[count]);
	}


	return 0;
}
//===============================================================================
int hierarchicalSegmentation::supervoxelAtTreeNode(TreeNode<nodeHierarchicalSegmentation>* hsNode, supervoxel& sv)
{
	
	//find all teh descendants that are leave nodes
	vector<supervoxel*> svVec;
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( hsNode );
	TreeNode<nodeHierarchicalSegmentation>* auxNode;

	while( q.empty() == false )
	{
		auxNode = q.front();
		q.pop();

		if( auxNode->getNumChildren() == 0 )
		{
			//leaf node
			svVec.push_back(auxNode->data.svPtr);
		}else{
			q.push( auxNode->left );
			q.push( auxNode->right );
		}
	}

	//merge all supervoxels
	sv.PixelIdxList.clear();
	if( svVec.empty() == false )
	{
		sv.TM = svVec[0]->TM;
		sv.dataPtr = svVec[0]->dataPtr;
		sv.localBackgroundSubtracted = svVec[0]->localBackgroundSubtracted;
	}
	sv.mergeSupervoxels( svVec);

	sv.nodeHSptr = hsNode;
	

	return 0;
}

//===============================================================================
size_t hierarchicalSegmentation::supervoxelAtTreeNodeOnlySize(TreeNode<nodeHierarchicalSegmentation>* hsNode)
{
	
	size_t sizeSv = 0;
	//find all teh descendants that are leave nodes
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( hsNode );
	TreeNode<nodeHierarchicalSegmentation>* auxNode;

	while( q.empty() == false )
	{
		auxNode = q.front();
		q.pop();

		if( auxNode->getNumChildren() == 0 )
		{
			//leaf node
			sizeSv += auxNode->data.svPtr->PixelIdxList.size();
		}else{
			q.push( auxNode->left );
			q.push( auxNode->right );
		}
	}
	
	return sizeSv;
}


//===================================================================
int hierarchicalSegmentation::debugCheckDendrogramCoherence()
{
	cout<<"DEBUGGING: hierarchicalSegmentation::debugCheckDendrogramCoherence()==========================="<<endl;
	//traverse tree to set id
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( dendrogram.pointer_mainRoot() );

	TreeNode<nodeHierarchicalSegmentation>* aux;
	int count = 0;
	while( !q.empty() )
	{
		aux = q.front();
		q.pop();

		if( aux->left != NULL )
		{
			q.push( aux->left );

			if( aux->left->data.thrTau > aux->data.thrTau )
			{
				cout<<"ERROR: at debugCheckDendrogramCoherence: left son has larger tauThr than parent. Parent*"<<aux->left->parent<<";aux*="<<aux<<";count="<<count<<endl;
				return 3;
			}

			if( aux->left->parent != aux )
			{
				cout<<"ERROR: at debugCheckDendrogramCoherence: parent left son do not agree. Parent*"<<aux->left->parent<<";aux*="<<aux<<";count="<<count<<endl;
				return 3;
			}
		}
		if( aux->right != NULL )
		{
			q.push( aux->right );
			
			if( aux->right->data.thrTau > aux->data.thrTau )
			{
				cout<<"ERROR: at debugCheckDendrogramCoherence: right son has larger tauThr than parent. Parent*"<<aux->right->parent<<";aux*="<<aux<<";count="<<count<<endl;
				return 3;
			}

			if( aux->right->parent != aux )
			{
				cout<<"ERROR: at debugCheckDendrogramCoherence: parent right son do not agree"<<endl;
				return 4;
			}
		}

		if( aux->getNumChildren() == 1 )
		{
			cout<<"WARNING: at debugCheckDendrogramCoherence: node has only one child; count="<<count<<endl;
		}

		if( aux->getNumChildren() == 0 )//this should be a basic region
		{
			if( aux->data.svPtr == NULL )
			{
				cout<<"ERROR: at debugCheckDendrogramCoherence: basic region has no pointer to supervoxel"<<endl;
				return 4;
			}
			
			//try to find it in basicRegions
			bool found = false;
			for(unsigned int aa = 0; aa < numBasicRegions; aa++ )
			{
				if( &(basicRegionsVec[aa]) == aux->data.svPtr )
				{
					found = true;
					break;
				}

			}
			if( found == false)
			{
				cout<<"ERROR: at debugCheckDendrogramCoherence: basic region has pointer to supervoxel but it is not in the list of basicRegionsVEc"<<endl;
				return 4;
			}
		}else{
			if( aux->data.svPtr != NULL )
			{
				cout<<"WARNING: at debugCheckDendrogramCoherence: none basic region has pointer to supervoxel"<<endl;				
			}
		}

		count++;
	}

	return 0;
}

//=====================================================================================
void hierarchicalSegmentation::findVeryPersistantBasicRegions( vector<supervoxel*>& svVec)
{
	svVec.clear();

	//traverse tree to set id using breadth first
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( dendrogram.pointer_mainRoot() );

	TreeNode<nodeHierarchicalSegmentation>* aux;
	int count = 0;
	while( !q.empty() )
	{
		aux = q.front();
		q.pop();


		if( aux->getNumChildren() == 0 && aux->data.svPtr != NULL)
		{
			if( aux->parent->data.thrTau >= maxTau )
				svVec.push_back( aux->data.svPtr);
		}
		else{
			if( aux->left != NULL )
			{
				q.push( aux->left );			
			}
			if( aux->right != NULL )
			{		
				q.push( aux->right );
			}
		}
		
	}
}

//======================================================================================
void hierarchicalSegmentation::findRealisticNodes(unsigned int minSize)
{
	//reset vectors
	currentSegmentationNodes.clear();

	//explore the tree using breadth first and keep all the nodes that are below maxTau and the ones that have a minSize. More details in notebook February 14th 2013
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( dendrogram.pointer_mainRoot() );
	TreeNode<nodeHierarchicalSegmentation>* auxNode;

	unsigned int chSizeA, chSizeB;

	while ( q.empty() == false )
	{
		auxNode = q.front();
		q.pop();

		if( auxNode->getNumChildren() == 0 )//basic region
		{
			if( auxNode->data.svPtr->PixelIdxList.size() > minSize )
				currentSegmentationNodes.push_back( auxNode );
		}else{

			//check daughters size
			chSizeA = supervoxelAtTreeNodeOnlySize( auxNode->left);
			chSizeB = supervoxelAtTreeNodeOnlySize( auxNode->right);
			

			//it should have two children always (it is a dendrogram)			
			if( auxNode->data.thrTau < maxTau )
			{
				currentSegmentationNodes.push_back( auxNode );	
			}
			
			if( chSizeA > minSize || chSizeB > minSize )
			{
				q.push( auxNode->left );
				q.push( auxNode->right );
			}
		}
	}

	//generate all the supervoxels for each segmentation node selected
	currentSegmentatioSupervoxel.resize( currentSegmentationNodes.size() );
	int count = 0;
	for(vector< TreeNode<nodeHierarchicalSegmentation>* >::iterator iter = currentSegmentationNodes.begin(); iter != currentSegmentationNodes.end(); ++iter, ++count)
	{
		supervoxelAtTreeNode((*iter),currentSegmentatioSupervoxel[count]);
	}
}

//===============================================================================================
template<class imgTypeC>
void hierarchicalSegmentation::cleanHierarchyWithTrimming(unsigned int minSize, unsigned int maxSize, int devCUDA)
{
	//preset parameters
	float thrJduplicates = 0.1;//to remove duplicates after trimming
	float maxPrctile = 0.4;//at the most keeping maxPrctile of teh pixels in supervoxel
	int conn3Dtrim = 6;
	int conn3DIsNeigh = 74;
	int64 boundarySizeSv[dimsImage];	
	int64* neighOffsetTrim = supervoxel::buildNeighboorhoodConnectivity(conn3Dtrim, boundarySizeSv);
	int64 boundarySizeIsNeigh[dimsImage];	
	int64* neighOffsetIsNeigh = supervoxel::buildNeighboorhoodConnectivity(conn3DIsNeigh + 1, boundarySizeIsNeigh);//using the special neighborhood for coarse sampling

	cout<<"A: Number of realistic nodes with maxTau "<<maxTau<<" and minSize "<<minSize<<"...";
	findRealisticNodes(minSize);
	cout<<" is "<<currentSegmentatioSupervoxel.size()<<endl;
	
	//trim each supervoxel and remove duplicates(it happens after trimming)
	cout<<"Trimming each possible supervoxel "<<endl;
	const int sizeW = (1+dimsImage) * dimsImage /2;
	for(vector<supervoxel>::iterator iter = currentSegmentatioSupervoxel.begin(); iter != currentSegmentatioSupervoxel.end(); ++iter)
	{
		iter->trimSupervoxel<imgTypeC>(maxSize,maxPrctile, conn3Dtrim, neighOffsetTrim);
	}
	

	//----------------------------------------------------------------------------------
	//remove parents in which the two children are not touching after the trimming	
	cout<<"Removing parents in which children do not neighbor after trimming"<<endl;
	TreeNode<nodeHierarchicalSegmentation> *ch1, *ch2;
	TreeNode<nodeHierarchicalSegmentation>* auxNode;
	int ch1Idx, ch2Idx;
	vector< TreeNode<nodeHierarchicalSegmentation>* >::iterator it;
	vector< int > eraseIdx;
	eraseIdx.reserve (currentSegmentationNodes.size() / 10);
	for( int ii = currentSegmentationNodes.size() - 1; ii >= 0 ; ii--)//we start from bottom to top (breadth first, so leaves are at the end) so we can remove quickly all the ancestors
	{
		if( currentSegmentationNodes[ii]->getNumChildren() == 0 )
			continue;

		if( find(eraseIdx.begin(), eraseIdx.end(), ii ) != eraseIdx.end() )//node already marked for deletion
			continue;

		ch1 = currentSegmentationNodes[ii]->left;
		ch2 = currentSegmentationNodes[ii]->right;

		//check if all of them are present: it was constructed using breadth first -> I do not need to search all the vector
		it = find(currentSegmentationNodes.begin() + ii, currentSegmentationNodes.end(), ch1);
		if( it == currentSegmentationNodes.end() )//not found
			continue;
		else
			ch1Idx = it - currentSegmentationNodes.begin();
		it = find(currentSegmentationNodes.begin() + ii, currentSegmentationNodes.end(), ch2);
		if( it == currentSegmentationNodes.end() )//not found
			continue;
		else
			ch2Idx = it - currentSegmentationNodes.begin();

		//check if parent needs to be erased
		if( currentSegmentatioSupervoxel[ ch1Idx ].isNeighboring( currentSegmentatioSupervoxel[ ch2Idx ],conn3DIsNeigh, neighOffsetIsNeigh ) == false )
		{
			eraseIdx.push_back( ii );
			//remove all the ancestors of ii
			auxNode = currentSegmentationNodes[ii];
			while( auxNode->parent != NULL )
			{
				auxNode = auxNode->parent;
				it = find(currentSegmentationNodes.begin(), currentSegmentationNodes.begin() + ii, auxNode);//breadth first search
				if( it != ( currentSegmentationNodes.begin() + ii) )//node found
					eraseIdx.push_back( it - currentSegmentationNodes.begin() );
			}
		}
	}
	//keep unique elements in order so they are easy to erase
	sort(eraseIdx.begin(), eraseIdx.end());
	vector<int >::iterator it2 = std::unique (eraseIdx.begin(), eraseIdx.end());                                                       
	eraseIdx.resize( std::distance(eraseIdx.begin(),it2) ); 
	cout<<"Erasing "<<eraseIdx.size()<<" parent nodes with not touching childrend out of "<<currentSegmentationNodes.size()<<" nodes..."<<endl;;
	//better to swap and then resize
	supervoxel auxSv;		
	size_t posSwap = currentSegmentatioSupervoxel.size() - 1;
	for(int ii = eraseIdx.size() - 1; ii>= 0; ii--, posSwap--)
	{
		auxSv = currentSegmentatioSupervoxel[ eraseIdx[ii] ];
		currentSegmentatioSupervoxel[ eraseIdx[ii] ] = currentSegmentatioSupervoxel[ posSwap ];
		currentSegmentatioSupervoxel[ posSwap ] = auxSv;
		auxNode = currentSegmentationNodes[ eraseIdx[ii] ];
		currentSegmentationNodes[ eraseIdx[ii] ] = currentSegmentationNodes[ posSwap ];
		currentSegmentationNodes[ posSwap ] = auxNode;
	}
	currentSegmentationNodes.resize( currentSegmentationNodes.size() - eraseIdx.size() );
	currentSegmentatioSupervoxel.resize( currentSegmentatioSupervoxel.size() - eraseIdx.size() );


	//----------------------------------------------------------------------------------
	//remove the ones that after trimming are too similar (keep the one with lowest tau)
	cout<<"Removing supervoxels that are the same after trimming"<<endl;
	eraseIdx.clear();
	vector<imgTypeC> JvecDuplicates;
	JvecDuplicates.reserve( 3 *  currentSegmentatioSupervoxel.size() );
	

	for(vector<supervoxel>::iterator iter = currentSegmentatioSupervoxel.begin(); iter != currentSegmentatioSupervoxel.end(); ++iter)
	{	
		iter->weightedCentroid<imgTypeC>();//here we only need centroid for the nearest neighbors
	}

	eraseIdx.push_back(0);//just to enter teh for loop
	while( eraseIdx.empty() == false)//some elements could have more than maxKNN to erase
	{
		vector< vector< vector<supervoxel>::iterator > > nearestNeighborVec;
		int errK = supervoxel::nearestNeighbors(currentSegmentatioSupervoxel, currentSegmentatioSupervoxel, supervoxel::getmaxKNNCUDA() , 5.0, devCUDA, nearestNeighborVec);//only elements very near by
		if ( errK > 0 )
			exit(errK);
		float auxJ;
		int neighIdx;
		eraseIdx.clear();
		for(size_t ii = 0; ii < nearestNeighborVec.size(); ii ++)
		{
			if( find(eraseIdx.begin(), eraseIdx.end(), ii) != eraseIdx.end() )
				continue;//we have already marked this element to delete
			for(size_t jj = 0; jj < nearestNeighborVec[ii].size(); jj++)
			{
				if( nearestNeighborVec[ii][jj] == (currentSegmentatioSupervoxel.begin() + ii) )
					continue;//make sure we are not looking at ourselves
				auxJ = currentSegmentatioSupervoxel[ii].JaccardDistance( *(nearestNeighborVec[ii][jj]) );
				JvecDuplicates.push_back( auxJ );
				if( auxJ < thrJduplicates)
				{
					neighIdx = nearestNeighborVec[ii][jj] - currentSegmentatioSupervoxel.begin();
					//mark for erasing 
					if( currentSegmentationNodes[ii]->data.thrTau > currentSegmentationNodes[ neighIdx ]->data.thrTau )//erase the one with highest tau
					{
						eraseIdx.push_back(ii);
						break;//we have already marked ii
					}
					else
						eraseIdx.push_back(neighIdx);
				}
			}
		}
		cout<<"Erasing "<<eraseIdx.size()<<" parent nodes that are the same out of "<<currentSegmentationNodes.size()<<" nodes..."<<endl;
		//keep unique elements in order so they are easy to erase
		sort(eraseIdx.begin(), eraseIdx.end());
		vector<int >::iterator it = std::unique (eraseIdx.begin(), eraseIdx.end());                                                       
		eraseIdx.resize( std::distance(eraseIdx.begin(),it) ); 
		
		//better to swap and then resize
		supervoxel auxSv;
		TreeNode<nodeHierarchicalSegmentation>* auxNode;
		size_t posSwap = currentSegmentatioSupervoxel.size() - 1;
		for(int ii = eraseIdx.size() - 1; ii>= 0; ii--, posSwap--)
		{
			auxSv = currentSegmentatioSupervoxel[ eraseIdx[ii] ];
			currentSegmentatioSupervoxel[ eraseIdx[ii] ] = currentSegmentatioSupervoxel[ posSwap ];
			currentSegmentatioSupervoxel[ posSwap ] = auxSv;
			auxNode = currentSegmentationNodes[ eraseIdx[ii] ];
			currentSegmentationNodes[ eraseIdx[ii] ] = currentSegmentationNodes[ posSwap ];
			currentSegmentationNodes[ posSwap ] = auxNode;
		}
		currentSegmentationNodes.resize( currentSegmentationNodes.size() - eraseIdx.size() );
		currentSegmentatioSupervoxel.resize( currentSegmentatioSupervoxel.size() - eraseIdx.size() );
	}




	//release memory
	delete[] neighOffsetIsNeigh;
	delete[] neighOffsetTrim;
}


//===============================================================================================
template<class imgTypeC>
void hierarchicalSegmentation::cleanHierarchy()
{
	//preset parameters
	float thrJduplicates = 0.1;//to remove duplicates after trimming
	int conn3DIsNeigh = 74;	
	int64 boundarySizeIsNeigh[dimsImage];	
	int64* neighOffsetIsNeigh = supervoxel::buildNeighboorhoodConnectivity(conn3DIsNeigh + 1, boundarySizeIsNeigh);//using the special neighborhood for coarse sampling
	

	if( dendrogram.pointer_mainRoot() == NULL || numBasicRegions <= 0 )
		return;

	//mark the basic regions
	for( unsigned int ii = 0; ii<numBasicRegions; ii++)
	{
		basicRegionsVec[ii].tempWildcard = 1.0f;
	}

	//we proceed top to bottom for rule 0: disregard any node above maxTau
	//we look for the nodes that are below maxTau for teh first time and add them as new roots
	deque< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push_back( dendrogram.pointer_mainRoot() );
	TreeNode<nodeHierarchicalSegmentation>* auxNode, *root, *oldRoot, *par;

	/* In reality, since maxTau is a variable part of the class, we do not need to lose this information. We can just propagate it trhough the code for merge / split. Anything above maxTau is like Infinity
	oldRoot = dendrogram.pointer_mainRoot();
	while ( q.empty() == false )
	{
		auxNode = q.front();
		q.pop_front();

		if( auxNode->data.thrTau < maxTau )//we select this region and we do not look at descendants
		{

			root = new TreeNode<nodeHierarchicalSegmentation>();
			root->parent = NULL;//new dendrogram root
			root->left = oldRoot;
			oldRoot->parent = root;
			oldRoot = root;//change root

			root->right = auxNode;
			par = auxNode->parent;
			if( par->left == auxNode )
				par->left = NULL;
			else
				par->right = NULL;
			auxNode->parent = root;

			root->data.svPtr = NULL;
			root->data.thrTau = std::numeric_limits<imgVoxelType>::max();

			//delete parent recursively if there are no more children left
			while( par!= NULL && par->getNumChildren() == 0 )
			{
				root = par;
				par = par->parent;
				delete root;				
			}

		}else{//we keep traversing this branch
			if( auxNode->left != NULL )
				q.push_back( auxNode->left );
			if( auxNode->right != NULL )
				q.push_back( auxNode->right );

		}

	}
	dendrogram.SetMainRootToNULL();
	dendrogram.SetMainRoot( oldRoot );
	*/


	//apply rules 1 and 2
	//rule 1: remove parents where Jaccard( parent, ch_A) ~= 1
	//rule 2: separate both children if they do not touch after trimming
	TreeNode<nodeHierarchicalSegmentation> *chR, *chL;
	q.push_back( dendrogram.pointer_mainRoot() );

	root = dendrogram.pointer_mainRoot();	
	oldRoot = dendrogram.pointer_mainRoot();

	int TM = basicRegionsVec[0].TM;
	void *dataPtr = basicRegionsVec[0].dataPtr;
	supervoxel *chLsv, *chRsv;//auxiliary variables

	bool auxNodeReinserted = false;

	int countJL = 0;
	int countJR = 0;
	int countNeigh = 0;
	int countExploredNodes = 0;
	while( q.empty() == false )
	{
		auxNode = q.front();
		q.pop_front();

		chL = auxNode->left;
		chR = auxNode->right;
		if( chL != NULL )
			q.push_back( chL );
		
		if( chR != NULL )
			q.push_back( chR );

		
		if( chR == NULL && chL == NULL )
		{
			if( auxNode->data.svPtr != NULL && auxNode->data.svPtr->tempWildcard < 0.0f )
				delete auxNode->data.svPtr;
			continue; //it cannot be a basic region		
		}
		
		if( auxNode->data.thrTau >= maxTau )
		{
			if( auxNode->data.svPtr != NULL && auxNode->data.svPtr->tempWildcard < 0.0f )
				delete auxNode->data.svPtr;
			continue; //this is already a blocked merging
		}
		//trim supervoxels that have not been trimmed
		if( auxNode->data.svPtr == NULL )
		{			
			auxNode->data.svPtr = new supervoxel( TM );
			auxNode->data.svPtr->dataPtr = dataPtr;
			auxNode->data.svPtr->tempWildcard = -1.0f;//not a basic region
			supervoxelAtTreeNode( auxNode, *(auxNode->data.svPtr) );
			auxNode->data.svPtr->trimSupervoxel<imgTypeC>();
		}
		//trim also children
		if( chL->data.svPtr == NULL )
		{			
			chL->data.svPtr = new supervoxel( TM );
			chL->data.svPtr->dataPtr = dataPtr;
			chL->data.svPtr->tempWildcard = -1.0f;//not a basic region
			supervoxelAtTreeNode( chL, *(chL->data.svPtr) );
			chL->data.svPtr->trimSupervoxel<imgTypeC>();
			chLsv = chL->data.svPtr;
		}else if( chL->data.svPtr->tempWildcard > 0.0f )
		{
			//basic region->I cannot trim directly in here
			chLsv = new supervoxel( *(chL->data.svPtr) );
			chLsv->trimSupervoxel<imgTypeC>();
			chLsv->tempWildcard = 1.0f;
		}else//precomputed
			chLsv = chL->data.svPtr;
		if( chR->data.svPtr == NULL )
		{			
			chR->data.svPtr = new supervoxel( TM );
			chR->data.svPtr->dataPtr = dataPtr;
			chR->data.svPtr->tempWildcard = -1.0f;//not a basic region
			supervoxelAtTreeNode( chR, *(chR->data.svPtr) );
			chR->data.svPtr->trimSupervoxel<imgTypeC>();
			chRsv = chR->data.svPtr;
		}else if( chR->data.svPtr->tempWildcard > 0.0f )
		{
			//basic region->I cannot trim directly in here
			chRsv = new supervoxel( *(chR->data.svPtr) );			
			chRsv->trimSupervoxel<imgTypeC>();
			chRsv->tempWildcard = 1.0f;
		}else//precomputed
			chRsv = chR->data.svPtr;

		countExploredNodes++;
		//rule 1: remove parents where Jaccard( parent, ch_A) ~= 1
		if( auxNode->data.svPtr->JaccardDistance( *(chLsv) ) < thrJduplicates )
		{
			countJL++;
			//set chR in new root
			root = new TreeNode<nodeHierarchicalSegmentation>();
			root->parent = NULL;//new dendrogram root
			root->left = oldRoot;
			oldRoot->parent = root;
			oldRoot = root;//change root

			root->right = chR;
			chR->parent = root;

			root->data.svPtr = NULL;
			root->data.thrTau = std::numeric_limits<imgVoxelType>::max();

			//add children of chL as children of auxNode
			if( chL->left != NULL )
			{
				auxNode->left = chL->left;
				auxNode->left->parent = auxNode;
			}else{
				auxNode->left = NULL;
			}
			if( chL->right != NULL )
			{
				auxNode->right = chL->right;
				auxNode->right->parent = auxNode;
			}else{
				auxNode->right = NULL;
			}

			//delete chL (also from queue)
			if( q.back() == chL )
				q.pop_back();
			else{
				par = q.back();
				q.pop_back();
				if( q.back() == chL )
					q.pop_back();
				q.push_back( par );
			}

			if( chLsv->tempWildcard > 0.0f)//basic region about ot be deleted->auxNode now is basic region with the same components as chL
			{
				auxNode->data.svPtr = chL->data.svPtr;
			}
			else if( chLsv != NULL)
				delete chL->data.svPtr;

			delete chL;
			chL = NULL;
			//I need to analize auxNode again since its children have changed
			q.push_back( auxNode );	
			auxNodeReinserted = true;
				

		}else if( auxNode->data.svPtr->JaccardDistance( *(chRsv) ) < thrJduplicates )
		{
			countJR++;
			//set chL in new root
			root = new TreeNode<nodeHierarchicalSegmentation>();
			root->parent = NULL;//new dendrogram root
			root->left = oldRoot;
			oldRoot->parent = root;
			oldRoot = root;//change root

			root->right = chL;
			chL->parent = root;

			root->data.svPtr = NULL;
			root->data.thrTau = std::numeric_limits<imgVoxelType>::max();

			//add children of chR as children of auxNode
			if( chR->left != NULL )
			{
				auxNode->left = chR->left;
				auxNode->left->parent = auxNode;
			}else{
				auxNode->left = NULL;
			}
			if( chR->right != NULL )
			{
				auxNode->right = chR->right;
				auxNode->right->parent = auxNode;
			}else{
				auxNode->right = NULL;
			}

			//delete chR (also from queue)
			if( q.back() == chR )
				q.pop_back();
			else{
				par = q.back();
				q.pop_back();
				if( q.back() == chR )
					q.pop_back();
				q.push_back( par );
			}

			if( chRsv->tempWildcard > 0.0f)//basic region about ot be deleted->auxNode now is basic region with the same components as chR
			{
				auxNode->data.svPtr = chR->data.svPtr;
			}else if( chRsv != NULL)
				delete chR->data.svPtr;
			delete chR;
			chR = NULL;

			//I need to analize auxNode again since its children have changed
			q.push_back( auxNode );
			auxNodeReinserted = true;

		}else if( chLsv->isNeighboring(*(chRsv), conn3DIsNeigh, neighOffsetIsNeigh) == false)
		{//rule 2: separate both children if they do not touch after trimming
			countNeigh++;
			root = new TreeNode<nodeHierarchicalSegmentation>();
			root->parent = NULL;//new dendrogram root
			root->left = oldRoot;
			oldRoot->parent = root;
			oldRoot = root;//change root

			root->right = auxNode;
			par = auxNode->parent;
			if( par->left == auxNode )
				par->left = NULL;
			else
				par->right = NULL;
			auxNode->parent = root;
			auxNode->data.thrTau = std::numeric_limits<imgVoxelType>::max();

			root->data.svPtr = NULL;
			root->data.thrTau = std::numeric_limits<imgVoxelType>::max();
		}
		//release memory
		if( auxNode->data.svPtr->tempWildcard < 0.0f && auxNodeReinserted == false)
		{
			delete auxNode->data.svPtr;//we are doing top to bottom->parent is not needed anymore
			auxNode->data.svPtr = NULL;
		}
		if( chL != NULL && chL->data.svPtr->tempWildcard > 0.0f )
			delete chLsv;
		if( chR != NULL && chR->data.svPtr->tempWildcard > 0.0f )
			delete chRsv;
	}
	dendrogram.SetMainRootToNULL();
	dendrogram.SetMainRoot( oldRoot );

	
	
	//clean nodes that have only one child
	q.push_back( dendrogram.pointer_mainRoot() );
	root = dendrogram.pointer_mainRoot();

	while( q.empty() == false )
	{
		auxNode = q.front();
		q.pop_front();

		if( auxNode->getNumChildren() == 1)
		{
			if( auxNode->left != NULL )
				chL = auxNode->left;
			else
				chL = auxNode->right;

			if( chL->getNumChildren() == 0 )//basic region->auxNode becomes basic region
			{
				auxNode->data.svPtr = chL->data.svPtr;
				if( auxNode->left != NULL )
					auxNode->left = NULL;
				else
					auxNode->right = NULL;
				delete chL;

			}else{//no basic region->inherit children from chL
				auxNode->left = chL->left;
				if(chL->left != NULL )
					chL->left->parent = auxNode;
				auxNode->right = chL->right;
				if(chL->right != NULL )
					chL->right->parent = auxNode;
				delete chL;
			}
		}

		if( auxNode->left != NULL )
			q.push_back(auxNode->left);
		if( auxNode->right != NULL )
			q.push_back(auxNode->right);
	}
	
	//release memory
	delete[] neighOffsetIsNeigh;

	cout<<"countJR = "<<countJR<<"; countJL = "<<countJL<<"; countNeigh = "<<countNeigh<<" after exploring "<<countExploredNodes<<endl;
	
	if( debugCheckDendrogramCoherence() > 0)
	{
		cout<<"========WARNING: hierarchicalSegmentation::cleanHierarchy: still has some minor issues. Look Notebook April 13th 2013==========================="<<endl;
	}
}



//=======================================================================
//\param[in]  eraseIdx: it has to be SORTED (ASCENDING) UNIQUE set of indexes to erase
void hierarchicalSegmentation::eraseSupervoxelFromCurrentSegmentation(const vector<int>& eraseIdx)
{
	if( eraseIdx.empty() == true )
		return;
	if( eraseIdx.back() >= currentSegmentatioSupervoxel.size() )
	{
		cout<<"ERROR: at hierarchicalSegmentation::eraseSupervoxelFromCurrentSegmentation: last eraseIdx[ii] is larger than currentSegmentatioSupervoxel size (out of range)"<<endl;
		exit(3);
	}

	//better to swap and then resize
	supervoxel auxSv;		
	TreeNode<nodeHierarchicalSegmentation>* auxNode;
	size_t posSwap = currentSegmentatioSupervoxel.size() - 1;
	for(int ii = eraseIdx.size() - 1; ii>= 0; ii--, posSwap--)
	{
		auxSv = currentSegmentatioSupervoxel[ eraseIdx[ii] ];
		currentSegmentatioSupervoxel[ eraseIdx[ii] ] = currentSegmentatioSupervoxel[ posSwap ];
		currentSegmentatioSupervoxel[ posSwap ] = auxSv;
		auxNode = currentSegmentationNodes[ eraseIdx[ii] ];
		currentSegmentationNodes[ eraseIdx[ii] ] = currentSegmentationNodes[ posSwap ];
		currentSegmentationNodes[ posSwap ] = auxNode;
	}
	currentSegmentationNodes.resize( currentSegmentationNodes.size() - eraseIdx.size() );
	currentSegmentatioSupervoxel.resize( currentSegmentatioSupervoxel.size() - eraseIdx.size() );
}


//=======================================================================================================
//check notebook April 14th 2013 for more details
template<class imgTypeC>
float hierarchicalSegmentation::suggestMerge(TreeNode<nodeHierarchicalSegmentation>* root, supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>** rootMerge,  supervoxel& rootMergeSv, int debugRecursion)
{
	const float thrJduplicates = 0.1;//to decide extreme cases

	//int64 boundarySizeIsNeigh[dimsImage];
	//int conn3DIsNeigh = 74;
	//int64* neighOffsetIsNeigh = supervoxel::buildNeighboorhoodConnectivity(conn3DIsNeigh + 1, boundarySizeIsNeigh);//using the special neighborhood for coarse sampling

	//reset solution
	*rootMerge = NULL;
	rootMergeSv.PixelIdxList.clear();

	//simple checks
	if( root == NULL )
		return 0.0f;

	*rootMerge = root->parent;
	
	if( *rootMerge == NULL )
		return 0.0f;

	if( (*rootMerge)->data.thrTau > maxTau )
	{
		//cout<<"Not merge because is above maxTau="<<maxTau<<endl;
		*rootMerge = NULL;
		return 0.0f;
	}


	//check to make sure after trimming they are not the same region
	supervoxelAtTreeNode(*rootMerge, rootMergeSv);//generate supervoxel
	rootMergeSv.trimSupervoxel<imgTypeC>();//trim supervoxel
	

	
	if( rootSv.JaccardDistance( rootMergeSv ) < thrJduplicates )
	{
		TreeNode<nodeHierarchicalSegmentation>* rootAux = (*rootMerge);
		supervoxel rootSvAux(rootMergeSv);

		//==============================================
		/*
		PROCESS_MEMORY_COUNTERS pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
		cout<<"DEBUGGING: suggestMerge: about to call recursion. debugRecursion = "<<debugRecursion<<". Total memory used = "<<pmc.WorkingSetSize / pow(2.0,20) <<"MB"<<endl;
		*/
		//===========================================
		return suggestMerge<imgTypeC>(rootAux, rootSvAux,  rootMerge, rootMergeSv, debugRecursion + 1);
	}

	
	//check if the other child is the same as the parent
	TreeNode<nodeHierarchicalSegmentation>* sibiling;
	supervoxel sibilingSv(rootSv.TM);

	if( (*rootMerge)->left == root )
	{
		sibiling = (*rootMerge)->right;
	}else{
		sibiling = (*rootMerge)->left;
	}

	supervoxelAtTreeNode(sibiling, sibilingSv);//generate supervoxel
	sibilingSv.trimSupervoxel<imgTypeC>();//trim supervoxel
	
	
	if( sibilingSv.JaccardDistance( rootMergeSv ) < thrJduplicates )
	{
		//cout<<"Not merge because is sibling is the same as parent"<<endl;
		//we cannot merge
		rootMergeSv.PixelIdxList.clear();
		*rootMerge = NULL;
		return 0.0f;
	}
	


	//check that the sizes match
	if( fabs( ((float)rootMergeSv.PixelIdxList.size()) - ((float)sibilingSv.PixelIdxList.size()) - ((float)rootSv.PixelIdxList.size()) ) / ( (float) rootMergeSv.PixelIdxList.size() ) > 0.1 )
	{
		//we cannot merge
		rootMergeSv.PixelIdxList.clear();
		*rootMerge = NULL;
		return 0.0f;
	}

	//cout<<"We pass all the tests, so we can merge with parent"<<endl;
	
	//note: I do not return the sibiling since I need to find all the current segmentation elements that are descendants of the rootMerge node. Use function findDescendantsInCurrentSegmentation(...)
	return rootMergeSv.mergePriorityFunction<imgTypeC>(rootSv,sibilingSv);//probability of the merge

}

//==================================================================================
//suggestSplit and suggestMerge might not be reversible if daughters do not touch. But that is normal since if they do not touch, it is a sure split.
//Check Notebook April 17th 2013 for more details. 
template<class imgTypeC>
float hierarchicalSegmentation::suggestSplit(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>* rootSplit[2],  supervoxel rootSplitSv[2])
{
	const float thrJduplicates = 0.1;//to decide extreme cases

	//reset solution
	rootSplit[0] = NULL;
	rootSplit[1] = NULL;
	rootSplitSv[0].PixelIdxList.clear();
	rootSplitSv[1].PixelIdxList.clear();

	//simple checks
	if( root->getNumChildren() != 2)
		return 0.0f;

	//check Jaccard difference
	rootSplit[0] = root->left;
	rootSplit[1] = root->right;
	for( int aa = 0; aa <2; aa++)
	{
		supervoxelAtTreeNode(rootSplit[aa], rootSplitSv[aa]);//generate supervoxel
		rootSplitSv[aa].trimSupervoxel<imgTypeC>();//trim supervoxel

		if( rootSv.JaccardDistance( rootSplitSv[aa] ) < thrJduplicates )
		{
			TreeNode<nodeHierarchicalSegmentation>* rootAux = rootSplit[aa];
			supervoxel rootSvAux(rootSplitSv[aa]);
			return suggestSplit<imgTypeC>(rootAux, rootSvAux, rootSplit, rootSplitSv);
		}
	}

	/*
	//check that the sizes match
	if( fabs( ((float)rootSv.PixelIdxList.size()) - ((float)rootSplitSv[0].PixelIdxList.size()) - ((float)rootSplitSv[1].PixelIdxList.size()) ) / ( (float) rootSv.PixelIdxList.size() ) > 0.1 )
	{
		//we cannot merge
		rootMergeSv.PixelIdxList.clear();
		*rootMerge = NULL;
		return 0.0f;
	}
	*/


	//we pass all the tests, so we can split
	
	return (1.0f - rootSv.mergePriorityFunction<imgTypeC>(rootSplitSv[0], rootSplitSv[1]) );//1 - probability of the merge = probability of split
}


//===================================================================================
int hierarchicalSegmentation::debugTestMergeSplitSuggestions(imgVoxelType tau)
{

	cout<<"DEBUG: hierarchicalSegmentation::debugTestMergeSplitSuggestions: EXPECTING SUPERVOXELS IN UINT16 DATA!!!!"<<endl;

	
	//perform basic segmentation
	segmentationAtTau(tau);

	//for each element try to suggest a merge
	TreeNode< nodeHierarchicalSegmentation > *rootMerge = NULL;
	supervoxel rootMergeSv;
	TreeNode< nodeHierarchicalSegmentation > *rootSplit[2];
	supervoxel rootSplitSv[2];

	float score;
	for(size_t ii = 0 ;ii < currentSegmentatioSupervoxel.size(); ii++ )
	{
		//trim supervoxel
		currentSegmentatioSupervoxel[ii].trimSupervoxel<unsigned short int>();

		score = suggestMerge<unsigned short int>(currentSegmentationNodes[ii], currentSegmentatioSupervoxel[ii], &rootMerge, rootMergeSv);

		if( rootMerge != NULL && score > 0)
		{
			//find split and make sure one of the splits is the original root
			suggestSplit<unsigned short int>(rootMerge, rootMergeSv, rootSplit, rootSplitSv );

			if( rootSplit[0] != currentSegmentationNodes[ii] && (rootSplit[1] != currentSegmentationNodes[ii]) )
			{
				bool isJequivalent = false;
				for(int aa = 0; aa < 2; aa++)
				{
					//check if they are equivalent in Jaccard sense
					supervoxel svAux;
					supervoxelAtTreeNode(rootSplit[aa],svAux);
					svAux.trimSupervoxel<unsigned short int>();
					float J = currentSegmentatioSupervoxel[ii].JaccardDistance( svAux);
					if( J < 0.1 )
					{
						isJequivalent = true;
						break;
					}
				}
				if( isJequivalent == false )
					cout<<"WARNING: at node "<<ii<<" of current segmentation split/merge are not reversible"<<endl;
			}

			//save suggestion in lineage to isually check
			cout<<"RESULT: at node "<<ii<<" of current segmentation split/merge is reversible"<<endl;
		}
	
	}


	return 0;
}


//=======================================================================================
void hierarchicalSegmentation::findDescendantsInCurrentSegmentation(TreeNode<nodeHierarchicalSegmentation>* root, vector< TreeNode<nodeHierarchicalSegmentation>* >& vecD)
{
	vecD.clear();
	if( root == NULL ) 
		return;

	TreeNode<nodeHierarchicalSegmentation>* par;
	for(size_t ii = 0; ii < currentSegmentationNodes.size(); ii++)
	{
		par = currentSegmentationNodes[ii];

		while( par != NULL )
		{
			if( par == root )
			{
				vecD.push_back( currentSegmentationNodes[ii] );
				break;
			}

			par = par->parent;
		}
	}
}

//================================================================================
int hierarchicalSegmentation::debugNumberOfNodesBelowTauMax()
{
	int nn = 0;
	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	TreeNode<nodeHierarchicalSegmentation>* aux;
	
	if ( dendrogram.pointer_mainRoot() == NULL )
		return nn;

	q.push( dendrogram.pointer_mainRoot() );

	while(q.empty() == false )
	{
		aux = q.front();
		q.pop();

		if( aux->left != NULL )
			q.push( aux->left );
		if( aux->right != NULL )
			q.push( aux->right );

		if( aux->data.thrTau < maxTau )
			nn++;
	}

	return nn;
}


//======================================================================================
void hierarchicalSegmentation::debugHierarchyDepth(string filename)
{
	cout<<"DEBUG::hierarchicalSegmentation::debugHierarchyDepth: writing file to " << filename << endl;
	
	ofstream fout(filename.c_str() );

	queue< TreeNode<nodeHierarchicalSegmentation>* > q;
	q.push( dendrogram.pointer_mainRoot() );
	TreeNode<nodeHierarchicalSegmentation>* auxNode;


	while(q.empty() == false )
	{
		auxNode = q.front();
		q.pop();

		if( auxNode->data.thrTau < maxTau ) //first of its kind->check depth
		{
			int numN = 0;
			queue< TreeNode<nodeHierarchicalSegmentation>* > qN;
			qN.push( auxNode );

			while(qN.empty() == false )
			{
				auxNode = qN.front();
				qN.pop();
				if( auxNode->left != NULL )
					qN.push( auxNode->left );
				if( auxNode->right != NULL )
					qN.push( auxNode->right );
				numN++;
			}

			fout<<numN<<endl;
		}else{//keep adding children
			if( auxNode->left != NULL )
				q.push( auxNode->left );
			if( auxNode->right != NULL )
				q.push( auxNode->right );
		}
	}

	fout.close();
}


//========================================================================
void hierarchicalSegmentation::debugEstimateDeltaZsupervoxel(imgVoxelType tau, string filename)
{
	cout<<"DEBUGGING:hierarchicalSegmentation::debugEstimateDeltaZsupervoxel: writing results to "<<filename<<endl;
	
	float alpha = 0.05;//percentile to make it robust estimation

	segmentationAtTau(tau);
	ofstream fout(filename);
	uint64 zTop, zBottom;
	uint64 coord[dimsImage];
	for(size_t ii = 0; ii < currentSegmentatioSupervoxel.size(); ii++)
	{
		int sizeSv = currentSegmentatioSupervoxel[ii].PixelIdxList.size();

		uint64 pos = currentSegmentatioSupervoxel[ii].PixelIdxList[(int)(alpha * sizeSv)];
		supervoxel::getCoordinates(pos, coord);
		zBottom = coord[ dimsImage - 1]; 

		pos = currentSegmentatioSupervoxel[ii].PixelIdxList[(int)((1.0f-alpha) * sizeSv)];
		supervoxel::getCoordinates(pos, coord);
		zTop = coord[ dimsImage - 1]; 

		fout<<zTop - zBottom<<endl;
	}

	fout.close();
	return;
}

//========================================================================
void  hierarchicalSegmentation::resetNodeIdDendrogram(vector<TreeNode<nodeHierarchicalSegmentation>*> &mapNodId2ptr)
{
	mapNodId2ptr.clear();
	mapNodId2ptr.reserve( 2* numBasicRegions );
	
	queue< TreeNode<nodeHierarchicalSegmentation>* > q; 
	q.push( dendrogram.pointer_mainRoot() );

	TreeNode<nodeHierarchicalSegmentation>* auxNode;
	int nodeId = 0;
	while( q.empty() == false )
	{
		auxNode = q.front();
		q.pop();

		if( auxNode->left != NULL )
		{
			q.push( auxNode->left );
		}
		if( auxNode->right != NULL )
		{
			q.push( auxNode->right );
		}
		auxNode->nodeId = nodeId;
		mapNodId2ptr.push_back( auxNode );
		nodeId++;
	}
}


//=========================================================
//template declaration
template void hierarchicalSegmentation::cleanHierarchyWithTrimming<float>( unsigned int minSize, unsigned int maxSize, int devCUDA);
template void hierarchicalSegmentation::cleanHierarchyWithTrimming<unsigned short int>( unsigned int minSize, unsigned int maxSize, int devCUDA);
template void hierarchicalSegmentation::cleanHierarchyWithTrimming<unsigned char>( unsigned int minSize, unsigned int maxSize, int devCUDA);

template void hierarchicalSegmentation::cleanHierarchy<float>();
template void hierarchicalSegmentation::cleanHierarchy<unsigned short int>();
template void hierarchicalSegmentation::cleanHierarchy<unsigned char>();

template float hierarchicalSegmentation::suggestMerge<float>(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>** rootMerge,  supervoxel& rootMergeSv, int debugRecursion);
template float hierarchicalSegmentation::suggestMerge<unsigned short int>(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>** rootMerge,  supervoxel& rootMergeSv, int debugRecursion);
template float hierarchicalSegmentation::suggestMerge<unsigned char>(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>** rootMerge,  supervoxel& rootMergeSv, int debugRecursion);

template float hierarchicalSegmentation::suggestSplit<float>(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>* rootSplit[2],  supervoxel rootSplitSv[2]);
template float hierarchicalSegmentation::suggestSplit<unsigned short int>(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>* rootSplit[2],  supervoxel rootSplitSv[2]);
template float hierarchicalSegmentation::suggestSplit<unsigned char>(TreeNode<nodeHierarchicalSegmentation>* root,  supervoxel& rootSv, TreeNode<nodeHierarchicalSegmentation>* rootSplit[2],  supervoxel rootSplitSv[2]);


