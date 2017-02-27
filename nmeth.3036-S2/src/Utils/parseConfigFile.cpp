/*
 * parseConfigFile.cpp
 *
 *  Created on: Oct 19, 2011
 *      Author: amatf
 */

#include "parseConfigFile.h"
#include <cstring>

const int MAX_CHARS_PER_LINE = 1024;
const int MAX_TOKENS_PER_LINE = 2;//right now we only expect configVariable:value
const char* DELIMITER = "=";


int configOptionsTrackingGaussianMixture::parseConfigFileTrackingGaussianMixture(const string &filename)
{

	ifstream configFile(filename.c_str());
	if(!configFile.good())
	{
		cout<<"ERROR at parseConfigFileTrackingGaussianMixture: config filename "<<filename<<" could not be opened"<<endl;
		return 1;
	}

	int n;
	while (!configFile.eof())
	{
		// read an entire line into memory
		char buf[MAX_CHARS_PER_LINE];
		configFile.getline(buf, MAX_CHARS_PER_LINE);


		if(strncmp(buf,"#",1)==0) continue;//comment

		// parse the line into DELIMITER-delimited tokens

		// array to store memory addresses of the tokens in buf
		char* token[MAX_TOKENS_PER_LINE];

		// parse the line
		token[0] = strtok(buf, DELIMITER); // first token
		n=0;
		if (token[0]!=NULL) // zero if line is blank
		{
			for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
			{
				token[n] = strtok(NULL, DELIMITER); // subsequent tokens
				if (token[n]==NULL) break; // no more tokens
			}
		}

		if(n!=2) continue;

		if(strcmp("imgFilePattern",token[0])==0)
		{
			imgFilePattern=string(token[1]);
		}else if(strcmp("debugPathPrefix",token[0])==0)
		{
			debugPathPrefix=string(token[1]);		
		}else if(strcmp("offsetFile",token[0])==0)
		{
			offsetFile = string(token[1]);		
		}else if(strcmp("thrPercentile",token[0])==0)
		{
			thrPercentile=atof(token[1]);
		}else if(strcmp("maxIterEM",token[0])==0)
		{
			maxIterEM=atoi(token[1]);
		}else if(strcmp("tolLikelihood",token[0])==0)
		{
			tolLikelihood=atof(token[1]);
		}else if(strcmp("betaPercentageOfN_k",token[0])==0)
		{
			betaPercentageOfN_k=atof(token[1]);
		}else if(strcmp("nuPercentageOfN_k",token[0])==0)
		{
			nuPercentageOfN_k=atof(token[1]);
		}else if(strcmp("alphaPercentage",token[0])==0)
		{
			alphaPercentage=atof(token[1]);
		}else if(strcmp("thrSplitScore",token[0])==0)
		{
			thrSplitScore=atof(token[1]);
		}else if(strcmp("thrCellDivisionPlaneDistance",token[0])==0)
		{
			thrCellDivisionPlaneDistance=atof(token[1]);
		}
		/*
		else if(strcmp("useIlastikBackgroundDetector",token[0])==0)
		{
			if(strcmp("false",token[1])==0)
				useIlastikBackgroundDetector=false;
			else
				useIlastikBackgroundDetector=true;
		}
		else if(strcmp("thrSignal",token[0])==0)
		{
			thrSignal=atof(token[1]);
		}*/
		else if(strcmp("estimateOpticalFlow",token[0])==0)
		{
			estimateOpticalFlow=atoi(token[1]);
		}else if(strcmp("maxDistPartitionNeigh",token[0])==0)
		{
			maxDistPartitionNeigh=atof(token[1]);
		}else if(strcmp("deathThrOpticalFlow",token[0])==0)
		{
			deathThrOpticalFlow=atoi(token[1]);
		}else if(strcmp("persistanceSegmentationTau",token[0])==0)
		{
			tau=atoi(token[1]);
		}else if(strcmp("maxNumKNNsupervoxel",token[0])==0)
		{
			KmaxNumNNsupervoxel = atoi(token[1]);
		}else if(strcmp("maxDistKNNsupervoxel",token[0])==0)
		{
			KmaxDistKNNsupervoxel=atof(token[1]);
		}else if(strcmp("temporalWindowForLogicalRules",token[0])==0)
		{
			temporalWindowRadiusForLogicalRules = atoi(token[1]);
		}else if(strcmp("anisotropyZ",token[0])==0)
		{
			anisotropyZ = atof(token[1]);
		}else if(strcmp("thrBackgroundDetectorHigh",token[0])==0)
		{
			thrBackgroundDetectorHigh = atof(token[1]);
		}else if(strcmp("thrBackgroundDetectorLow",token[0])==0)
		{
			thrBackgroundDetectorLow = atof(token[1]);
		}else if(strcmp("SLD_lengthTMthr",token[0])==0)
		{
			SLD_lengthTMthr = atoi(token[1]);
		}else if(strcmp("radiusMedianFilter",token[0])==0)
		{
			radiusMedianFilter = atoi(token[1]);
		}else if(strcmp("minTau",token[0])==0)
		{
			minTau = atof(token[1]);
		}else if(strcmp("backgroundThreshold",token[0])==0)
		{
			backgroundThreshold = atof(token[1]);
		}else if(strcmp("conn3D",token[0])==0)
		{
			conn3D = atoi(token[1]);
		}else if(strcmp("regularizePrecisionMatrixConstants_lambdaMin",token[0])==0)
		{
			lambdaMin = atof(token[1]);
		}else if(strcmp("regularizePrecisionMatrixConstants_lambdaMax",token[0])==0)
		{
			lambdaMax = atof(token[1]);
		}else if(strcmp("regularizePrecisionMatrixConstants_maxExcentricity",token[0])==0)
		{
			maxExcentricity = atof(token[1]);
		}else if(strcmp("minNucleiSize",token[0])==0)
		{
			minNucleiSize = atoi(token[1]);
		}else if(strcmp("maxNucleiSize",token[0])==0)
		{
			maxNucleiSize = atoi(token[1]);
		}else if(strcmp("maxPercentileTrimSV",token[0])==0)
		{
			maxPercentileTrimSV = atof(token[1]);
		}else if(strcmp("conn3DsvTrim",token[0])==0)
		{
			conn3DsvTrim = atoi(token[1]);
		}else{
			cout<<"WARNING at parseConfigFileTrackingGaussianMixture: does not recognize config option "<<token[0]<<endl;
		}		
}

	configFile.close();

	//check that the mandatory values are not set to default
	if(strcmp(imgFilePattern.c_str(),"empty")==0)
	{
		cout<<"ERROR at parseConfigFileTrackingGaussianMixture: mandatory variable imgFilePattern was not set at config filename "<<filename<<" could not be opened"<<endl;
		return 1;
	}

	if(strcmp(debugPathPrefix.c_str(),"empty")==0)
	{
		cout<<"ERROR at parseConfigFileTrackingGaussianMixture: mandatory variable debugPathPrefix was not set at config filename "<<filename<<" could not be opened"<<endl;
		return 1;
	}

	if( backgroundThreshold < -1e31 )
	{
		cout<<"ERROR at parseConfigFileTrackingGaussianMixture: mandatory variable backgroundThreshold was not set at config filename "<<filename<<" could not be opened"<<endl;
		return 1;
	}

	return 0;
}


void configOptionsTrackingGaussianMixture::printConfigFileTrackingGaussianMixture(ostream &outLog)
{
	outLog<<"imgFilePattern="<<imgFilePattern<<endl;
	outLog<<"maxIterEM="<<maxIterEM<<endl;
	outLog<<"tolLikelihood="<<tolLikelihood<<endl;
	outLog<<"betaPercentageOfN_k="<<betaPercentageOfN_k<<endl;
	outLog<<"nuPercentageOfN_k="<<nuPercentageOfN_k<<endl;
	outLog<<"alphaPercentage = "<<alphaPercentage<<endl;
	outLog<<"thrSplitScore="<<thrSplitScore<<endl;
	outLog<<"thrCellDivisionPlaneDistance="<<thrCellDivisionPlaneDistance<<endl;
	outLog<<"thrPrcentile="<<thrPercentile<<endl;
	//outLog<<"thrSignal="<<thrSignal<<endl;
	//outLog<<"Use Ilastik background/foreground detection="<<useIlastikBackgroundDetector<<endl;
	outLog<<"estimateOpticalFlow="<< estimateOpticalFlow<<endl;
	outLog<<"deathThrOpticalFlow="<< deathThrOpticalFlow<<endl;
	outLog<<"maxDistPartitionNeigh="<< maxDistPartitionNeigh<<endl;
	outLog<<"persistanceSegmentationTau="<< tau<<endl;
	outLog<<"maxNumKNNsupervoxel="<<KmaxNumNNsupervoxel<<endl;
	outLog<<"maxDistKNNsupervoxel="<<KmaxDistKNNsupervoxel<<endl;
	outLog<<"temporalWindowForLogicalRules (radius)="<<temporalWindowRadiusForLogicalRules<<endl;
	outLog<<"anisotropyZ="<<anisotropyZ<<endl;
	outLog<<"offset file = "<<offsetFile<<endl;

	outLog<<"thrBackgroundDetectorHigh="<<thrBackgroundDetectorHigh<<endl;
	outLog<<"thrBackgroundDetectorLow="<<thrBackgroundDetectorLow<<endl;
	outLog<<"short-lived duaghter thre = "<<SLD_lengthTMthr<<endl;

	outLog<<"radiusMedianFilter ="<<radiusMedianFilter<<endl;
	outLog<<"minTau ="<<minTau<<endl;
	outLog<<"backgroundThreshold ="<<backgroundThreshold<<endl;
	outLog<<"conn3D ="<<conn3D<<endl;

	outLog<<"regularizePrecisionMatrixConstants_lambdaMin = "<<lambdaMin<<endl;
	outLog<<"regularizePrecisionMatrixConstants_lambdaMax = "<<lambdaMax<<endl;
	outLog<<"regularizePrecisionMatrixConstants_maxExcentricity = "<<maxExcentricity<<endl;


	outLog<<"Trim supervoxel minNucleiSize = "<<minNucleiSize<<endl;
	outLog<<"Trim supervoxel maxNucleiSize ="<<maxNucleiSize<<endl;
	outLog<<"Trim supervoxel maxPercentileTrimSV = "<<maxPercentileTrimSV<<endl;
	outLog<<"Trim supervoxel conn3DsvTrim = "<<conn3DsvTrim<<endl;
}
