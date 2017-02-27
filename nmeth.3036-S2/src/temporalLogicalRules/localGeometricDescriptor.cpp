/*
 * Copyright (C) 2011-2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  localGeometricDescriptor.cpp
 *
 *  Created on: July 21st, 2013
 *      Author: Fernando Amat
 *
 * \brief Local geometric descriptors for point clouds
 *        
 *
 */

#include <iostream>
#include <algorithm>
#include <limits>
#include <math.h>
#include "localGeometricDescriptor.h"

template<int D>
float localGeometricDescriptor<D>::neighRadius = 50.0f;
template<int D>
std::vector<float> localGeometricDescriptor<D>::refPts[maxTM_LGD][D];

using namespace std;


//====================================================================
template<int D>
localGeometricDescriptor<D>::localGeometricDescriptor()
{
};

//==================================================================
template<int D>
float localGeometricDescriptor<D>::distance(const localGeometricDescriptor &p, size_t k, float scale[D]) const
{
	//find the closest point from this to p
	vector<float> d(neighPts[0].size());
	float dAux;
	for(size_t ii = 0; ii < neighPts[0].size(); ii++)
	{
		float dMin = numeric_limits<float>::max();		
		for(size_t jj = 0; jj < p.neighPts[0].size(); jj++)//we xpect number of neighboring points to be small, so brute force search should be OK
		{
			dAux = 0.0f;
			for(int kk = 0; kk < D; kk++)
				dAux += (p.neighPts[kk][jj] - neighPts[kk][ii]) * (p.neighPts[kk][jj] - neighPts[kk][ii]) * scale[kk] * scale[kk];
			if ( dAux < dMin )
				dMin = dAux;
		}
		d[ii] = dMin;
	}

	//sort elements in ascending order
	sort(d.begin(), d.end());

	//calculate robust distance as RMS of the closest k-th values
	dAux = 0;
	for(size_t ii = 0; ii < min(k,d.size() ); ii++)
		dAux += d[ii];

	dAux = sqrt(dAux) / min(k,d.size());

	return dAux; //RMS of the k-th closest elements
}

//============================================================
template<int D>
void localGeometricDescriptor<D>::setNeighPts(const float p0[D], const vector<float> neighPts_[D])
{
	for(int ii = 0; ii < D; ii++)
	{
		neighPts[ii] = neighPts_[ii];
		for(size_t jj = 0; jj < neighPts[ii].size(); jj++)
		{
			neighPts[ii][jj] -= p0[ii];//translation invariant
		}
	}
}

//preinitialize here the number you might need for dimensions
template class localGeometricDescriptor<3>; //to precompile
