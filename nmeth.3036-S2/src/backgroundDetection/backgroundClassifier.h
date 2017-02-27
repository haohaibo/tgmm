/*
 * Copyright (C) 2011-2013 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *  backgroundClassifier.h
 *
 *  Created on: January 21st, 2013
 *      Author: Fernando Amat
 *
 * \brief features (temporal and geometrical and spatial) to distinguish background lineages
 *        
 *
 */

#ifndef __BACKGROUND_CLASSIFIER_H__
#define __BACKGROUND_CLASSIFIER_H__

#include <fstream>
#include "lineageHyperTree.h"

class backgroundDetectionFeatures
{
public:
	vector<float> f;


	//constructor / destructor
	backgroundDetectionFeatures();
	backgroundDetectionFeatures(int teporalWindowSize_);
	~backgroundDetectionFeatures();

	//set / get function
	static long long int getNumFeatures();
	static void setTemporalWindowSize(int p){ teporalWindowSize = p;};
	static int getTemporalWindowSize(void){ return teporalWindowSize;};

	//main function to calculate features
	/*
	\brief starting at the the given root it calculates a set of fatures. It uses teporalWindowSize to calculate the features

	\return			Greater than zero if features could not be calculated
	*/
	int calculateFeatures(TreeNode< ChildrenTypeLineage >* root, int devCUDA);

	//support functions
	static bool isCorrectConfidenceLevel(TreeNode< ChildrenTypeLineage >* root, int confidenceLevel);//whether we can use this lineage for training (looks at confidence number)
	void  writeToBinary(std::ofstream &fout);

protected:

private:

	static int teporalWindowSize;//number of time points includes to calculate features

};


//===========================================================================================
inline void  backgroundDetectionFeatures::writeToBinary(std::ofstream &fout)
{
	fout.write((char*)(&(f[0])), sizeof(float) * f.size() );
}

//======================================================================

inline long long int backgroundDetectionFeatures::getNumFeatures()
{
        return 35;//make sure this matches with backgroundDetectionFeatures::calculateFeatures (there is an assert)
}


#endif //__BACKGROUND_CLASSIFIER_H__
