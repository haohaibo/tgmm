/*
 * See license.txt for full license and copyright notice.
 *
 * \brief Calculates features for given lineage in a temporal interval in order to use them for different classification tasks
 *
 */



#ifndef __LINEAGE_WINDOW_FEATURES_H__
#define __LINEAGE_WINDOW_FEATURES_H__

#include <iostream>
#include "lineageHyperTree.h"



static const int lineageWindowNumFeatures = 10;//this number should match with getLineageWindowFeatures(...) function

//stores basic nucleus features that we can combine later in time
struct basicNucleusFeatures
{
	size_t size;//size of the nucleus (number of voxels belonging to it)
	float avgIntensity;//average intensity
	float numNeigh; //average number of neighbors that belong to the same nucleus (maximum is conn3D)
	float displacement; // displacement with respect to parent cell. 0 if it is root
	float ratioKNN;//distance ratio between nearest neighbor and k-furthest neighbor

	void reset()
	{
		size = 0;
		avgIntensity = 0.0f;
		numNeigh = 0.0f;
		//displacement = 0.0f;
		//ratioKNN = 0.0f;
	}

	basicNucleusFeatures& operator= (basicNucleusFeatures const& other);
};



/*
\brief Main function. Starting from root, it calculates features going down as much as winLength into the lineage. Feature vector should be independent of the lineage length (premature death) or the number of cell divisions (although topology information is included)

\param in root			Starting point to traverse the lineage to calculate features
\param in winLength		Length of the window to calculate features
\param out fVec			Vector containing all the feature values (it should be preallocated). Length = numfeatures
\return int				0 if no error. >0 id there is an error.
*/
template <class T>
int getLineageWindowFeatures(TreeNode<ChildrenTypeLineage>* root, unsigned int winLength,float* fVec);



/*
\brief: calculate features for a nucleus in individual time points. They can be geometric, image based or social features

\param in		nuc				nucleus to calculate all the features
\param in		conn3D			conn3D to calculate near neighbors for auxNEigh
\param out		data			workspace containing basic nucleus features
*/
//template T is the type of image data
template<class T>
void calculateNucleusTimeFeatures(nucleus &nuc, int conn3D, int64* neighOffset, basicNucleusFeatures &data);



//========================================================================
inline basicNucleusFeatures& basicNucleusFeatures::operator= (basicNucleusFeatures const& other)
{
	if(&other != this)
	{
		size = other.size;
		avgIntensity = other.avgIntensity;
		numNeigh = other.numNeigh;
		displacement = other.displacement;
		ratioKNN = other.ratioKNN;
	}
	return *this;
}


#endif //__LINEAGE_WINDOW_FEATURES_H__
