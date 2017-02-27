/*
 * constants.h
 *
 *  Created on: Jul 21, 2011
 *      Author: amatf
 *
 *      defines global constants across the program
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#ifndef DIMS_IMAGE_CONST //to protect agains teh same constant define in other places in the code
#define DIMS_IMAGE_CONST
static const int dimsImage=3; //image dimensions so we can precompile sizes of many arrays to make the code run faster
#endif


static const int maxGaussiansPerVoxel=5; //maximum number of Gaussians that can be considered for each voxel
static const double minPi_kForDead=1e-5;

struct regularizePrecisionMatrixConstants
{
	static double lambdaMin;//aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0 and maxRadius=10 (adjust with scale)
	static double lambdaMax;//aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0 and minRadius=3.0 (adjust with scale)  (when nuclei divide they can be very narrow)
	static double maxExcentricity;//maximum excentricity allowed: sigma[i]=1/sqrt(d[i]). Therefore maxExcentricity needs to be squared to used in terms of radius.

	regularizePrecisionMatrixConstants()
	{
		lambdaMin = -1.0;//so it can be checked if they ar einitialized or not
		lambdaMax = -1.0;
		maxExcentricity = -1.0;
	}

	static void setConstants(double lambdaMin_, double lambdaMax_, double maxExcentricity_)
	{
		lambdaMin = lambdaMin_;//so it can be checked if they ar einitialized or not
		lambdaMax = lambdaMax_;
		maxExcentricity = maxExcentricity_;
	}
};



#endif /* CONSTANTS_H_ */
