/*
 *      @brief Contains all the routines to perform varuational inference in one image
 */

#ifndef VARIATIONALINFERENCE_H_
#define VARIATIONALINFERENCE_H_

#include "GaussianMixtureModel.h"
#include "responsibilities.h"
#include <vector>
#include "UtilsCUDA/knnCuda.h"
#include "UtilsCUDA/GMEMupdateCUDA.h"

using namespace std;

namespace mylib
{
	extern "C"
	{
		#include "mylib/array.h"
	}
};

static const unsigned int maxNeighborsMRF = 10;
static const float maxDistThrMRF = 100.0f;

double updateResponsibilities(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r, double thrSignal);
void updateResponsibilitiesWithParticles(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r, double thrSignal);
void updateGaussianParameters(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r);
void updateGaussianParametersWithParticles(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r);

double calculateLogLikelihood(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r);
double calculateLogLikelihoodWithParticles(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r);
//model selection penalty based on
//[1] M. A. T. Figueiredo and A. K. Jain, ?Unsupervised Learning of Finite Mixture Models,? IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 24, p. 381?396, Mar. 2002.
double calculateModelSelectionPenalty(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r,double numSamples);

GaussianMixtureModel* splitGaussian(vector<GaussianMixtureModel*> &vecGM, responsibilities &r,int pos);
void deleteGaussian(vector<GaussianMixtureModel*> &vecGM, responsibilities &r,int pos);
void unfixedAllMixtures(vector<GaussianMixtureModel*> &vecGM);


//based on [1] N. Ueda, R. Nakano, Z. Ghahramani, and G. E. Hinton, ?SMEM Algorithm for Mixture Models,? Neural Computation, vol. 12, p. 2109?2128, Sep. 2000.
//returns a measure to estimate candidates to split. The higher the Kullback divergence value the more likely to split
void calculateLocalKullbackDiversity(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, const responsibilities &r);
//extension to pairs within the mixture
double calculatePairwiseLocalKullbackDiversity(mylib::Array *img,GaussianMixtureModel* GM1,GaussianMixtureModel* GM2, const responsibilities &r);

//apply flow prediction to center of mass
//imgFlowMask indicates where to apply the flow (it can be NULL to apply flow everywhere) 
int applyFlowPredictionToNuclei(mylib::Array* imgFlow,mylib::Array* imgFlowMask,vector<GaussianMixtureModel*> &vecGM,bool isForwardFlow);


//I/O functions
void copy2GMEM_CUDA(GaussianMixtureModel *GM,GaussianMixtureModelCUDA *GMCUDAtemp);
void copyFromGMEM_CUDA(GaussianMixtureModelCUDA *GMCUDAtemp,GaussianMixtureModel *GM);

//debug functions
bool checkVecGMidIntengrity(vector<GaussianMixtureModel*> &vecGM);

//======================================================================================================

//functions to calculate different expectations using particles
inline double pairwiseMRF(meanPrecisionSample &p,void *data)
{
	meanPrecisionSample *p2=(meanPrecisionSample*)data;
	double sigmaDist_o=p2->w;//void pointer stores sigma information
	double dist=GaussianMixtureModel::distEllipsoid(p,*p2)/sigmaDist_o;
	return(p.w*log(1.0-exp(-dist*dist)));
}
inline double logDeterminant(meanPrecisionSample &p,void *data)
{
	return p.w*log(p.lambda_k.determinant());
};
inline double mahalanobisDistance(meanPrecisionSample &p,void *data)
{
	Matrix<double,dimsImage,1> *x_n=(Matrix<double,dimsImage,1> *)data;
	return (p.w*(((*x_n)-p.mu_k).transpose()*p.lambda_k*((*x_n)-p.mu_k)))(0);
}

//auxiliary functions
inline bool isNaN(const double var)
{
	volatile double pV = var;//to avoid the chance that compiler -O3 ignores the statement
    return (pV != pV);
}

inline void unfixedAllMixtures(vector<GaussianMixtureModel*> &vecGM)
{
	for(vector<GaussianMixtureModel*>::iterator iter=vecGM.begin();iter!=vecGM.end();++iter) (*iter)->fixed=false;
}

#endif /* VARIATIONALINFERENCE_H_ */
