/*
 * parseConfigFile.h
 *
 *  Created on: Oct 19, 2011
 *      Author: amatf
 *
 *
 * /brief Reads and parse options from a config file. Each line
 *  in the file should be configOption:value. If the line starts 
 *  with # it is considered a comment
 */

#ifndef PARSECONFIGFILE_H_
#define PARSECONFIGFILE_H_

#include <string>
#include <iostream>
#include <fstream>

using namespace std;
//contains all the options to be handled by the program
struct configOptionsTrackingGaussianMixture
{

    //image filename is composed as imgPrefix+ii+imgSufix+ii+'.tif' 
    //where ii=frame number
	//string imgPrefix;
	//string imgSufix;

    //using ??? to adapt to any file notation and folder structure
	string imgFilePattern;

    //folder where to output solutions
	string debugPathPrefix;

    //optional parameter if time points need drift corrections
	string offsetFile;

    //xml file containing initialization for first frame
	//string GMxmlIniFilename;
    
	int maxIterEM;

    //we do not consider the priors so far to decide the
	//stopping criteria
	double tolLikelihood;

	//controls the strength of the priors from previous frames: it is a
    //proportion with respect to N_k.
    
    //uncertainty in center of the Gaussian. betaPercentageOfN_k<1.0 
    //indicates data dominates over prior to fit GM.
	//betaPercentageOfN_k>1.0 indicates prior dominates over data 
	//to fit GM.
	double betaPercentageOfN_k;

    //uncertainty in shape (covariance of the Gaussian)
	double nuPercentageOfN_k;

    //prior to let Gaussian die (the lower the value, the more 
    //likely to die). Relative to alphaTotal
	double alphaPercentage;

	//before we used super-voxels for local background subtraction
    //and oversegmentation
    
    //check September 16th notebook to see reasonable thresholds 
    //for adaBoost classifier
	double thrSplitScore;

    //inidicates if we want to use background/foreground detection 
    //from Ilastik as intensity map
	//bool useIlastikBackgroundDetector;

    //to remove false positive cell division detection
	double thrCellDivisionPlaneDistance;

    //(For Gene is in ascending order) We only keep 
	//thrPrecpercentile pixels.
    //percentile to threshold image to consider points that are signal
	double thrPercentile;

    //if useIlastikBackgroundDetector=true this value represents
    //the cut-off probability to consider as foreground. Otherwise, 
    //it is adjusted based on thrPrcentile
	//double thrSignal;

    //maximum number of nearest neighbors to consider for supervoxels
	int KmaxNumNNsupervoxel;

    //maximum distance (in pixels with scale) to consider a nearest 
    //neighbor between supervoxels
	float KmaxDistKNNsupervoxel;

    //we will use a sliding window of +- temporalWindowRadiusForLogicalRules 
    //to improve tracking precision
	int temporalWindowRadiusForLogicalRules;

	float anisotropyZ;

	//sporious track removal
	float thrBackgroundDetectorHigh;//parameters for background detection
	float thrBackgroundDetectorLow;
	int SLD_lengthTMthr;//short-lived daughter rul


	//watershed +PBC segmentation
	int		radiusMedianFilter;
	float	minTau;
	float	backgroundThreshold;
	int		conn3D;

	//optical flow parameters
    //0->do not use flow;1->flow is precacalculted (load file from disk);
    //2->calculate on the fly;
	int estimateOpticalFlow;
    //controls main regularization parameter for optical flow
	float maxDistPartitionNeigh;
    //if estimateOpticalFlow=0 we use this to selectively
    //activate if there are too many deaths in the process.
    //SET TO 0 IF YOU DON'T WANT THIS FEATURE
	int deathThrOpticalFlow;
	/*typical choices:
     * [2, 80.0, -20]->optical flow is calculated all the time; 
	   [0, 80.0,  20]->optical flow is only calculated if there 
                       are more than 20 deaths
	   [0, *   , -20]->optical flow is not used
	   [1, 80.0, -20]->pre-calculate image flow is read and used all the time
	*/

	int tau; //parameter for persistance segmentation

	//precision matrix regularization boundaries
    
    //aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0
    //and maxRadius=10 (adjust with scale)
	double lambdaMin;

    //aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0 and
    //minRadius=3.0 (adjust with scale)  (when nuclei divide they
    //can be very narrow)
	double lambdaMax;

    //maximum excentricity allowed: sigma[i]=1/sqrt(d[i]). Therefore 
    //maxExcentricity needs to be squared to used in terms of radius.
	double maxExcentricity;


	//------Trimming supervoxels using Otzu's threshold to
    //differentiate background from foreground------
    
	//Minimum size (in voxels) of a supervoxel
    //(smaller supervoxel will be deleted)
	int minNucleiSize;

	//Maximum size (in voxels) of a supervoxel
    //(considered when we apply Otzu's threshold)
	int	maxNucleiSize;
	//Maximum percentage of voxels in a supervoxel belonging to
    //foreground (considered when we apply Otzu's threshold)
	float	maxPercentileTrimSV;
	//Connectivity considered when trimming supervoxels 
	//using Otzu's threshold
	int	conn3DsvTrim;

	void setDefaultValues()
	{
		//these values are set to "non-sense" values since they
        //are mandatory in the config file
        
		//imgPrefix=string("empty");
		//imgSufix=string("empty");
		//GMxmlIniFilename=string("empty");
		imgFilePattern = string("empty");
		debugPathPrefix = string("empty");

		offsetFile = string("empty");

		//thrPercentile=-1.0; //from previous approach before super-voxels

		//most of these values should not be modified but user 
        //can decide to do it in the config file
		maxIterEM=400;
		tolLikelihood=1e-6;

		betaPercentageOfN_k=0.01;
		nuPercentageOfN_k=1.0;
		alphaPercentage = 0.8;

		thrSplitScore=-7.0;

		thrCellDivisionPlaneDistance = 3.14;
		
        //from previous approach before super-voxels
		//useIlastikBackgroundDetector=true; 
		//thrSignal=0.2;//from previous approach before super-voxels

		estimateOpticalFlow=0;//optical flow deactivated by default
		maxDistPartitionNeigh=80;
		deathThrOpticalFlow=-20;

		tau=0;

		KmaxNumNNsupervoxel = 10;
		KmaxDistKNNsupervoxel = 50.0f;
		temporalWindowRadiusForLogicalRules = 1;

		anisotropyZ = 1.0f;

		//above 1.0 background detection is not activated
		thrBackgroundDetectorHigh = 1.2;

		thrBackgroundDetectorLow = 0.2;
		SLD_lengthTMthr = 5;

		//elements for watershed segmentation
		radiusMedianFilter = 2;
		minTau = 2;
		backgroundThreshold = -1e32;
		conn3D = 74;

		//precision matrix regularization boundaries for GMM fitting
		lambdaMin = 0.02;//so it can be checked if they are einitialized or not
		lambdaMax = 0.2222;
		maxExcentricity = 3.0 * 3.0;

		//----- Trimming supervoxels using Otzu's threshold to differentiate background from foreground------
		//Minimum size (in voxels) of a supervoxel
        //(smaller supervoxel will be deleted)
		minNucleiSize = 50;
		//Maximum size (in voxels) of a supervoxel 
        //(considered when we apply Otzu's threshold)
		maxNucleiSize = 3000;
		//Maximum percentage of voxels in a supervoxel 
        //belonging to foreground (considered when we
        //apply Otzu's threshold)
		maxPercentileTrimSV = 0.4;
		//Connectivity considered when trimming supervoxels
        //using Otzu's threshold
		conn3DsvTrim = 6;
	}

	/*
	 *\brief returns 0 if everything was fine.returns>0 if there was an error
	 */
	int parseConfigFileTrackingGaussianMixture(const string &filename);
	void printConfigFileTrackingGaussianMixture(ostream &outLog);

    //default constructor
	configOptionsTrackingGaussianMixture(){setDefaultValues();};
};


#endif /* PARSECONFIGFILE_H_ */

