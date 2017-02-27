/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *
 * nuclei.h
 *
 *  Created on: August 17th, 2012
 *      Author: Fernando Amat
 *
 * \brief Simple class containing information for a nuclei. It is better to keep it separate and depending on the application the attributes you want to save are different
 *
 */


#ifndef __NUCLEI_TEMPORAL_LOGICAL_RULES_H__
#define __NUCLEI_TEMPORAL_LOGICAL_RULES_H__

#include <iostream>
#include <list>
#include <vector>
#include "../constants.h"
#include "edgeTree.h"


using namespace std;

//typedef declarations to build hierarchical tree
class supervoxel;//forward declaration
class lineage;
class nucleus;
template <class T> 
struct TreeNode;

typedef list< supervoxel >::iterator ChildrenTypeNucleus;
typedef list< lineage >::iterator ParentTypeNucleus;
typedef list< nucleus >::iterator SibilingTypeNucleus;
typedef list< nucleus >::iterator ChildrenTypeLineage;


//simple class to store nuclei detections
class nucleus
{
public:
	int TM;//time point that the nuclei belongs to
	float centroid[dimsImage];
	float avgIntensity;
	edgeTree< ParentTypeNucleus, ChildrenTypeNucleus> treeNode;
	TreeNode< ChildrenTypeLineage > *treeNodePtr;//points at the TreeNode within the lineage. NULL if it is not associated with any lineage
	//vector< SibilingTypeNucleus > sibilings;//for nearest neighbors of other nuclei. IT IS DONE THROUGH SUPERVOXELS
	float tempWilcard;//temporary storage space for different purposes. IT IS NOT COPIED WITH THE ASSIGNMENT OPERATOR OR THE COPY CONSTRUCTOR!!!

	float debugVisualization;//to mark specific nucleus in order to easily find them in the visualization

	float confidence;//parameter to match form CATMAID (or any other estimate of how confident we are about this nucleus);
	float probBackground;//score from background classifier. The higher, the more likely this nucleus is to be background

	//simple inline functions
	~nucleus();
	nucleus (const nucleus& p);
	nucleus(int time_,float* centroid_){ nucleusDefault(time_, centroid_);};
	nucleus()
	{
		float aux[dimsImage];
		//for(int ii = 0;ii<dimsImage;ii++) //no need to waste time
		//	aux[ii] = 0.0f;
		nucleusDefault(-1,aux);
		probBackground = -1.0f;
	}
	void nucleusDefault(int time_,float* centroid_);

	int addSupervoxelToNucleus(const ChildrenTypeNucleus &ch)//returns>0 if there was an error
	{
		treeNode.addChild(ch);
		return 0;
	}
	int removeSupervoxelFromNucleus(unsigned int idx)
	{
		return treeNode.deleteChild(idx);
	}
	int removeSupervoxelFromNucleus( ChildrenTypeNucleus& sv )
	{
		for( vector<ChildrenTypeNucleus>::iterator iter = treeNode.getChildren().begin(); iter != treeNode.getChildren().end(); ++iter)
		{
			if( sv == (*iter) )
			{
				treeNode.getChildren().erase( iter );
				return 0;
			}
		}
		return 1;//indicates it was not found
	};
	bool isDead()
	{
		return centroid[0] < -1e30f;
	}
	void setDead(){ centroid[0] = -1e32f;};

	//useful operations
	float Euclidean2Distance(const nucleus& s, float scale[dimsImage] ) const;

	//operators
	friend ostream& operator<< (ostream& os, nucleus m);
	nucleus& operator=(const nucleus & p);
	bool operator== (nucleus const& other) const; 
	bool isEqual(nucleus const& other) const;
	bool operator< (nucleus const& other) const; 
};


//=================================================
inline float nucleus::Euclidean2Distance(const nucleus& s, float scale[dimsImage]) const
{
	float dist = 0;
	float aux = 0;
	for(int ii=0;ii<dimsImage; ii++)
	{
		aux = (centroid[ii]-s.centroid[ii]) * scale[ii];
		dist += aux*aux;
	}
	return dist;
}

#endif