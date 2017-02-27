/*
 * responsibilities.h
 *
 *  Created on: May 12, 2011
 *      Author: amatf
 */

#ifndef RESPONSIBILITIES_H_
#define RESPONSIBILITIES_H_


#include "Utils/CSparse.h"
#include <iostream>
#include <vector>
#include "constants.h"

namespace mylib
{
extern "C"
{
#include "mylib/mylib.h"
#include "mylib/array.h"
#include "mylib/image.h"
#include "mylib/cdf.h"
}
};

using namespace std;


struct colormap
{
	double r,g,b;//rgb values for a colormap between [0,1.0]
	static void generateRandomColormap(vector<colormap> &colMap,unsigned int numColors)
	{
		colMap.clear();
		colMap.reserve(numColors);
		colormap aux;
		for(unsigned int ii=0;ii<numColors;ii++)
		{
			aux.r=mylib::drand();
			aux.g=mylib::drand();
			aux.b=mylib::drand();
			colMap.push_back(aux);
		}
	}

	static const float hsv18[54];//from Matlab hsv(18) saved into memory
};

class responsibilities
{
public:

	//main variables
	unsigned long long int N;//number of voxels per image
	unsigned long long int K;//number of clusters (or mixtures of Gaussians)
	cs *R_nk;//sparse responsibility matrix

	//constructor/destructor
	responsibilities(unsigned long long int N_,unsigned long long int K_,long long int nzmax);
	~responsibilities();

	//debug routines
	void writeOutMatlabFormat(string fileOut);
	void writeOutSegmentationMaskAndImageBlend(mylib::Array *img,string fileout,const vector<colormap> &colMap);

protected:

private:

};

void writeOutSegmentationMaskAndImageBlend(mylib::Array *img,string fileOut,const vector<colormap> &colMap,long long int *nPos,int *kAssignment,int *colorVec,long long int query_nb);

#endif /* RESPONSIBILITIES_H_ */
