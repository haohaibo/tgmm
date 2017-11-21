/*
 * See license.txt for full license and copyright notice.
 *
 * \brief 
 *
 */

#include <stdio.h>
#include <string.h>
#include <cstring>
#include "nuclei.h"


ostream& operator<<( std::ostream & os, const nucleus m )
{
	os<<"Centroid = ";
	for(int ii=0; ii<dimsImage; ii++)
		os<<m.centroid[ii]<<" ";
	os<<"; TM = "<<m.TM;

	return os;
};

//============================================================================
nucleus::~nucleus()
{
	treeNodePtr = NULL;
}
nucleus& nucleus::operator=(const nucleus& p)
{
	if (this != &p)
	{		
		TM = p.TM;
		memcpy(centroid,p.centroid,sizeof(float)*dimsImage);
		avgIntensity = p.avgIntensity;
		treeNode = p.treeNode;
		//sibilings = p.sibilings;
		treeNodePtr = p.treeNodePtr;

		debugVisualization = p.debugVisualization;
		confidence = p.confidence;
		probBackground = p.probBackground;

	}
	return *this;
};

//copy constructor
nucleus::nucleus (const nucleus& p)
{
	TM = p.TM;
	memcpy(centroid,p.centroid,sizeof(float)*dimsImage);
	avgIntensity = p.avgIntensity;
	treeNode = p.treeNode;
	//sibilings = p.sibilings;
	treeNodePtr = p.treeNodePtr;

	debugVisualization = p.debugVisualization;
	confidence = p.confidence;
	probBackground = p.probBackground;
};

//shared function by all constructors
void nucleus::nucleusDefault(int time_,float* centroid_)
{
	TM = time_;
	for(int ii = 0; ii<dimsImage;ii++)
		centroid[ii] = centroid_[ii];
	edgeTree<ParentTypeNucleus, ChildrenTypeNucleus> ();	
	treeNodePtr = NULL;

	debugVisualization = 0;
	probBackground = -1.0f;
}

//================================================================================
bool nucleus::operator< (nucleus const& other) const
{
	return (this->TM < other.TM);//I could expand it so if they belong to the same frame, use centroid
};

bool nucleus::operator== (nucleus const& other) const
{
	if(this->TM == other.TM)
	{
		float norm = 0.0f, aux;
		for(int ii=0;ii<dimsImage;ii++)
		{
			aux = (this->centroid[ii] - other.centroid[ii]);
			norm += (aux * aux);
		}
		if(norm < 1e-3)
			return true;
	}

	return false;//easy case
};
bool nucleus::isEqual (nucleus const& other) const
{
	if(this->TM == other.TM)
	{
		float norm = 0.0f, aux;
		for(int ii=0;ii<dimsImage;ii++)
		{
			aux = (this->centroid[ii] - other.centroid[ii]);
			norm += (aux * aux);
		}
		if(norm < 1e-3)
			return true;
	}

	return false;//easy case
};


