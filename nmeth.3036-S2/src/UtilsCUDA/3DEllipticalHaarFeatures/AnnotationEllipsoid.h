/*
 * Ellipsoid.h
 *
 *  Created on: Feb 15, 2011
 *      Author: amatf
 *
 *
 *      @brief  Ellipsoids in 3D are defined by a 3x3 covariance matrix. Therefore we only store the 6 lower triangular elements in the Surface::coeff vector in the following order
 *      for(int ii=0;ii<sigma1.cols();ii++)
		{
		for(int jj=ii;jj<sigma1.rows();jj++)
		{
			sigma1(jj,ii)=coeff[count++];
		}
		}

		coeff[6,7,8] contain origin (x,y,z)

		Given the covariance matrix, the major axis are the eigenvectors and the length of each axis is 1./sqrt(eigenvalues) . Always think of the decomposition (x-c)'A(x-c)=1 with A>=0 .
		thus A=V*D*V' as eigen decomposition -> [D^(1/2)*V'(x-c)]'* [D^(1/2)*V'(x-c)] = 1  decomposes rotation, centering and scaling very nicely
 */

#ifndef ELLIPSOID_H_
#define ELLIPSOID_H_

#include <string>
#include <iostream>
#include <vector>
#include "external/xmlParser/xmlParser.h"

#ifndef DIMS_IMAGE_CONST //to protect agains teh same constant define in other places in the code
	#define DIMS_IMAGE_CONST
	static const int dimsImage=3; //image dimensions so we can precompile sizes of many arrays to make the code run faster
#endif

using namespace std;

class AnnotationEllipsoid 
{
public:
	AnnotationEllipsoid();
	AnnotationEllipsoid(const double *mu_, const double *W_, string& imgFilename_);
	~AnnotationEllipsoid();
	AnnotationEllipsoid(const AnnotationEllipsoid &p);
	AnnotationEllipsoid (XMLNode &xml,int position);//to create object from reading XML file

	//operators
	AnnotationEllipsoid& operator=(const AnnotationEllipsoid& p);
	friend bool operator< (const AnnotationEllipsoid& lhs, const AnnotationEllipsoid& rhs);

	//main variables
	double mu[dimsImage];//centroid of the ellipsoid
	double W[ dimsImage * (1+ dimsImage) /2]; //covariance (symmetric matrix) of teh ellipsoid
	string imgFilename; //img filename where the object of inetrest is located
	string className;
	int classVal;//negative number indicates it has not been assigned yet in case of categorical values

	string svFilename;//binary filename containing the supervoxels
	vector<int> svIdx;//index of each of teh supervoxels that belong to this annotation. If empty, it means we do not have such information

	//short I/O functions	
	ostream& writeXML(ostream& os);

	
protected:

private:
	
};



#endif /* ELLIPSOID_H_ */
