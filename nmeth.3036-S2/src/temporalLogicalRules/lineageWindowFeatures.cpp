/*
 * Copyright (C) 2011-2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *
 * lineageWindowFeature.cpp
 *
 *  Created on: January 7th, 2013
 *      Author: Fernando Amat
 *
 * \brief Calculates features for given lineage in a temporal interval in order to use them for different classification tasks
 *
 */

#include <set>
#include <cmath>
#include <typeinfo>
#include <algorithm>
#include "lineageWindowFeatures.h"
#include "float.h"


//to port code to Unix systems
#if (!defined(_WIN32)) && (!defined(_WIN64))

#define _isnan(x) isnan(x)
#endif

using namespace std;

template <class T>
int getLineageWindowFeatures(TreeNode<ChildrenTypeLineage>* root, unsigned int winLength,float* fVec)
{

	int numFeatures = 0;
	const int conn3D = 6; //to calculate neighboring voxels with teh same id (background voxels tend to have less in average
	
	//reset fVec
	memset(fVec, 0, sizeof(float) * lineageWindowNumFeatures );

	if(root == NULL)
	{
		//cout<<"ERROR: at getLineageWindowFeatures: root is NULL"<<endl;
		return 0;
	}
	int TMmax = root->data->TM + winLength;//if aux->data->TM >= TMmax then we ignore it

	

	//first we need to build a list of teh different branches (to handle divisions)	
	int numCellDivisions = 0;
	int TMfinal = root->data->TM;//to see how long gets the furthest branch
	vector< TreeNode<ChildrenTypeLineage>* > vecRootBranches;//we only save the root of each branch. Follow it until we hit maxTM or we hit a cell division
	vecRootBranches.reserve(5);
	vecRootBranches.push_back(root);
	

	queue< TreeNode<ChildrenTypeLineage>* > q;
	TreeNode<ChildrenTypeLineage>* aux;
	q.push( root );
	
	while( q.empty() == false)
	{
		aux = q.front();
		q.pop();
		TMfinal = max( TMfinal, aux->data->TM );
		
		if( aux->data->TM >= TMmax)
			continue;//we have hit the maximum time point

		switch( aux->getNumChildren() )
		{
		case 1:
			if( aux->left != NULL)
			{
				q.push(aux->left);
			}else{
				q.push(aux->right);
			}
			break;

		case 2:
			numCellDivisions++;
			q.push(aux->left);
			q.push(aux->right);
			vecRootBranches.push_back(aux->left);
			vecRootBranches.push_back(aux->right);
			break;

		default:
			//do nothing;
			break;
		}
	}


	//toplogical features
	fVec[numFeatures++] = (float) (numCellDivisions);
	fVec[numFeatures] = (float) ( TMfinal + 1 - root->data->TM ) / ( (float) winLength);//proportion of length that we could reach


	TreeNode<ChildrenTypeLineage>* auxNext;
	float auxFl;
	int numChildren;

	basicNucleusFeatures auxFeatures, auxFeaturesNext;

	//set connectivity
	int64 boundarySize[dimsImage];
	int64* neighOffset = supervoxel::buildNeighboorhoodConnectivity(conn3D, boundarySize);

	int Npairwise = 0;//number of pairwise measurements (between adjacent time points)
	for(size_t ii = 0; ii < vecRootBranches.size(); ii++)
	{
		aux = vecRootBranches[ii];

		//calculate initial quantities to calculate delta increments
		calculateNucleusTimeFeatures<T>( *(aux->data), conn3D, neighOffset, auxFeatures);


		while( aux != NULL )
		{
			numChildren = aux->getNumChildren();

			if( numChildren == 1)
			{
				if(aux->left != NULL) 
					auxNext = aux->left;
				else
					auxNext = aux->right;

				//calculate basic nucleus features
				calculateNucleusTimeFeatures<T>( *(auxNext->data), conn3D, neighOffset, auxFeaturesNext);

				//calculate all the features that depend on pairwise
				Npairwise++;

				auxFl = ( auxFeatures.displacement - auxFeaturesNext.displacement ) / auxFeatures.displacement;//relative so it can be ported to other scenarios
				if( _isnan(auxFl) > 0 )
					auxFl = 0;//TODO: what is a good default value?
				fVec[numFeatures + 0] += auxFl;//geometric feature: average displacement L2 distance (without square root)
				fVec[numFeatures + 1] += (auxFl * auxFl);//geometric feature: std displacement L2 distance (without square root)
				
				auxFl = ( auxFeatures.avgIntensity - auxFeaturesNext.avgIntensity ) / auxFeatures.avgIntensity;//relative so it can be ported to other scenarios
				if( _isnan(auxFl) > 0 )
					auxFl = 0;//TODO: what is a good default value?
				fVec[numFeatures + 2] += auxFl;//image-based feature
				fVec[numFeatures + 3] += (auxFl * auxFl);//image-based fature

				auxFl = ( auxFeatures.numNeigh - auxFeaturesNext.numNeigh) / auxFeatures.numNeigh;//relative so it can be ported to other scenarios
				if( _isnan(auxFl) > 0 )
					auxFl = 0;//TODO: what is a good default value?
				fVec[numFeatures + 4] += auxFl;//image-based feature
				fVec[numFeatures + 5] += (auxFl * auxFl);//image-based feature
				
				auxFl = ( auxFeatures.size - auxFeaturesNext.size ) / ( (float) auxFeatures.size );//relative so it can be ported to other scenarios
				if( _isnan(auxFl) > 0 )
					auxFl = 0;//TODO: what is a good default value?
				fVec[numFeatures + 6] += auxFl;//geometric feature: //TODO: consider using SSE2 instructions to upload all these features at once
				fVec[numFeatures + 7] += (auxFl * auxFl);//geometric feature: 


				fVec[numFeatures + 8] += auxFeatures.ratioKNN;

				//cout<<"TDOO:THROW IN SOME OTHER FEATURES: SOME RINGS INTENSITY?SOCIAL FEATURES?!!!!"<<endl;
			}else{ 
				auxNext = NULL;
			}


			//select next element
			if( aux->data->TM < (TMmax-1) && numChildren == 1)
			{
				if(aux->left != NULL) 
					aux = aux->left;
				else
					aux = aux->right;

				//copy features
				auxFeatures = auxFeaturesNext;
			}else aux = NULL;
		}
	}

	//average all the quantities
	if ( Npairwise > 1)
	{
		fVec[ numFeatures + 0] /= ( (float) Npairwise );
		fVec[ numFeatures + 1] /= ( (float) Npairwise );
		fVec[ numFeatures + 1] -= ( fVec[ numFeatures + 0] * fVec[ numFeatures + 0]);//std is computed using 1/N instead of (1/ (N-1) ) for efficiency

		fVec[ numFeatures + 2] /= ( (float) Npairwise );
		fVec[ numFeatures + 3] /= ( (float) Npairwise );
		fVec[ numFeatures + 3] -= ( fVec[ numFeatures + 2] * fVec[ numFeatures + 2]);
		
		fVec[ numFeatures + 4] /= ( (float) Npairwise );
		fVec[ numFeatures + 5] /= ( (float) Npairwise );
		fVec[ numFeatures + 5] -= ( fVec[ numFeatures + 4] * fVec[ numFeatures + 4]);
		
		fVec[ numFeatures + 6] /= ( (float) Npairwise );
		fVec[ numFeatures + 7] /= ( (float) Npairwise );
		fVec[ numFeatures + 7] -= ( fVec[ numFeatures + 6] * fVec[ numFeatures + 6]);

		fVec[ numFeatures + 8] /= ( (float) Npairwise );
	}

	//update number of features
	numFeatures += 9;

	//checkthat we have allocate the right number of features
	if( numFeatures != lineageWindowNumFeatures )
	{
		cout<<"ERROR: at getLineageWindowFeatures: number of preallocated feature="<<lineageWindowNumFeatures<<";number of computed features ="<<numFeatures<<". It should be the same! Change const and recompile"<<endl;
		return 10;
	}


	delete[] neighOffset;

	return 0;
}

//define the templates allowed to avoid linker errors
template int getLineageWindowFeatures<unsigned char>(TreeNode<ChildrenTypeLineage>* root, unsigned int winLength,float* fVec);
template int getLineageWindowFeatures<unsigned short int>(TreeNode<ChildrenTypeLineage>* root, unsigned int winLength,float* fVec);
template int getLineageWindowFeatures<float>(TreeNode<ChildrenTypeLineage>* root, unsigned int winLength,float* fVec);

//==============================================================================================================
/*
\brief: calculate features for a nucleus in individual time points. They can be geometric, image based or social features

\param in		nuc				nucleus to calculate all the features
\param out		auxSize			size of the nucleus (number of voxels belonging to it)
\param out		auxIntensity	average intensity
\param out		auxNeigh		average number of neighbors that belong to the same nucleus (maximum is conn3D)

\param in		conn3D			conn3D to calculate near neighbors for auxNEigh
*/

template<class T>
void calculateNucleusTimeFeatures(nucleus &nuc, int conn3D, int64* neighOffset, basicNucleusFeatures &data)
{

	T *imgPtr;

	string type(typeid(T).name());

	if(type.compare("unsigned short")== 0)
	{
		if( supervoxel::getDataType() != 1 )
		{
			cout<<"ERROR: at calculateNucleusFeatures: type does not match with read data ptr"<<endl;
			exit(3);;
		}
	}else if( type.compare("float") == 0)
	{
		if( supervoxel::getDataType() != 8 )
		{
			cout<<"ERROR: at calculateNucleusFeatures: type does not match with read data ptr"<<endl;
			exit(3);
		}
	}else{
		cout<<"ERROR: at calculateNucleusFeatures: code is not ready for image data type "<<type<<endl;
		exit(3);
	}
	data.reset();

	vector<int> neighOffsetPosPos, neighOffsetNegPos;//to avoid if within for loop
	for(int jj = 0; jj < conn3D; jj++)
	{
		if(  neighOffset[jj] > 0 )
		{
			neighOffsetPosPos.push_back(jj);
		}else{
			neighOffsetNegPos.push_back(jj);
		}
	}


	for(vector<ChildrenTypeNucleus>::iterator iterS = nuc.treeNode.getChildren().begin(); iterS != nuc.treeNode.getChildren().end(); ++iterS)
	{
		data.size += (*iterS)->PixelIdxList.size(); //delta voulme of the nucleus
		imgPtr = (T*) ((*iterS)->dataPtr);
		
		for(vector<uint64>::iterator iterP = (*iterS)->PixelIdxList.begin(); iterP != (*iterS)->PixelIdxList.end(); ++iterP)
		{
			data.avgIntensity += imgPtr[ *iterP ];

			//check number of neighbors that are of the same label
			int auxN = 0;
			
			for(vector<int>::const_iterator iter = neighOffsetPosPos.begin(); iter != neighOffsetPosPos.end(); ++iter)
			{				
				if( binary_search( iterP , (*iterS)->PixelIdxList.end(), (*iterP) + neighOffset[*iter]) == true )
						auxN++;
			}
			for(vector<int>::const_iterator iter = neighOffsetNegPos.begin(); iter != neighOffsetNegPos.end(); ++iter)
			{
				if( binary_search( (*iterS)->PixelIdxList.begin(), iterP, (*iterP) + neighOffset[*iter]) == true )
						auxN++;
			}

			/*
			//this functions is really slow
			for(int jj = 0; jj < conn3D; jj++)
			{				
				if( (*iterS)->idxBelongsToPixelIdxListLinearSearch( (*iterP) + neighOffset[jj], (*iterP) ) == true )
					auxN++;								
			}
			*/

			data.numNeigh += ( (float) auxN ) / ( (float) conn3D );
		}

		//TODO: texture features;
	}

	data.numNeigh /= ( (float) data.size );
	data.avgIntensity /= ( (float) data.size );

	if( nuc.treeNodePtr->parent != NULL )
		data.displacement = nuc.Euclidean2Distance(*(nuc.treeNodePtr->parent->data),supervoxel::getScale());//geometric feature: L2 distance (without square root)
	else
		data.displacement = 0.0f;


	//find nearest neighbors and use it as a "social" feature
	unsigned int K = supervoxel::getKmaxNumNN() / 2;
	vector<float> distVec(K);
	vector<ChildrenTypeLineage> iterNucleusNNvec(K);

	int err = lineageHyperTree::findKNearestNucleiNeighborInSpaceSupervoxelEuclideanL2(nuc.treeNodePtr->data, iterNucleusNNvec, distVec);
	if( err > 0 )
		exit(err);

	data.ratioKNN = distVec.front() / distVec.back();//if there is no K nearest neighbor-> distVec.back() = 1e32 -> ratio = 0
}
