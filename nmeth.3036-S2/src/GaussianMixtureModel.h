/*
 * GaussianMixtureModel.cpp
 *
 *  Created on: May 12, 2011
 *      Author: amatf
 *
 *
 *      CheckBishop's book section 10.2 for more reference on notation
 */

#ifndef GAUSSIANMIXTUREMODEL_H_
#define GAUSSIANMIXTUREMODEL_H_

#include "external/Eigen/Core"
#include "external/Eigen/LU"
#include "external/Eigen/Eigenvalues"
#include "external/gsl/gsl_sf_psi.h" //to compute digamma function using GSL/GNU code
#include "external/xmlParser2/xmlParser.h"
#include <iostream>
#include <vector>
#include "constants.h"

static const int numParticles=1; //number of particles to characterize q(mu,sigma) distribution

using namespace Eigen;
using namespace std;

//------------------------------------------
//structure to store particles for the Q(mu,sigma) distribution
struct meanPrecisionSample
{
	Matrix<double,dimsImage,1> mu_k;//expected value for the mean of teh Gaussian
	Matrix<double,dimsImage,dimsImage> lambda_k;//expected value of the covariance of the Gaussian. It should be a symmetric matrix
	double w;//weight

	static const double nuProposal;
	static const double betaProposal;//needed for proposal distribution since the one obtained from data is too tight
};

//pointers to function to calculate expectations
typedef  double (*pfn_muLambdaExpectation)(meanPrecisionSample &p,void *data);

//definition for Kalman Filter: we implement a simple linear model with (position,velocity) as state
typedef  Matrix<double,2*dimsImage,1> VectorState;
typedef  Matrix<double,2*dimsImage,2*dimsImage> MatrixState;
typedef  Matrix<double,dimsImage,1> VectorObs;
typedef  Matrix<double,dimsImage,dimsImage> MatrixObs;
typedef  Matrix<double,dimsImage,2*dimsImage> MatrixState2Obs;



//------------------------------------------
class GaussianMixtureModel
{

public:

	int id;
	int lineageId;//to know parent (I need to add pointer to identity)
	int parentId;
	bool fixed;//fixed==true->we should not update parameters for this Gaussian mixture

	int color;//to paint reprsentations of the Gaussians

	//to store indexes of the supervoxel that the Gaussian belongs to
	vector<int> supervoxelIdx;

	//main variables to define a Gaussian in a mixture model
	Matrix<double,dimsImage,1> m_k;//expected value for the mean of teh Gaussian
	double beta_k;//proportionality constant to relate covariance matrix of the mean of the Gaussian and and the expected value of teh covariance itself
	Matrix<double,dimsImage,dimsImage> W_k;//expected value of the covariance of the Gaussian. It should be a symmetric matrix
	double nu_k;//degrees of freedom in the covariance. It has to be greater or equal than dimsImage
	double alpha_k;//responsivities. Expected value of the number of elements that belong to his Gaussian in the mixture model

	double N_k;//very important to setup priors order of magnitude
	double splitScore;//local Kullback divergence to consider splitting

	//---------------------------------------------------------------------------
	//priors
	Matrix<double,dimsImage,1> m_o;//expected value for the mean of teh Gaussian
	double beta_o;//proportionality constant to relate covariance matrix of the mean of the Gaussian and and the expected value of teh covariance itself
	Matrix<double,dimsImage,dimsImage> W_o;//expected value of the covariance of the Gaussian. It should be a symmetric matrix
	double nu_o;//degrees of freedom in the covariance. It has to be greater or equal than dimsImage
	double alpha_o;//responsivities. Expected value of the number of elements that belong to his Gaussian in the mixture model

	//MRF priors
	double sigmaDist_o;

	//particles (or samples) to characterize the distribution
	meanPrecisionSample muLambdaSamples[numParticles];

	//---------------------------------------------------------------------------
	//constructor/destructor
	GaussianMixtureModel();
	GaussianMixtureModel(int id_);
	GaussianMixtureModel(int id_,float scale_[dimsImage]);
	GaussianMixtureModel(XMLNode &xml,int position);
	GaussianMixtureModel(const GaussianMixtureModel & p);
	~GaussianMixtureModel();
	void resetValues();


	//needed for kdtree structure
	float dist;
	float center[dimsImage];
	float distBlobs(const float *cc);
	static float scale[dimsImage];//anisotropy in resolution between different dimensions

	//---------------------------------------------------------------------------
	//main routines to update parameters during variational inference
	double expectedLogDetCovariance(void);//equation 10.65
	double expectedLogResponsivity(double totalAlpha);//equation 10.66
	double expectedMahalanobisDistance(Matrix<double,dimsImage,1> &x_n);//equation 10.64
	void updateGaussianParameters(const Matrix<double,dimsImage,1> &x_k,const Matrix<double,dimsImage,dimsImage> &S_k);
	void regularizePrecisionMatrix(bool W4DOF);
	//------------------------------------------------------------------------
	//other methods
	void updateCenter(){for(int ii=0;ii<dimsImage;ii++) center[ii]=m_k(ii);};
	double distEllipsoid(GaussianMixtureModel &p);//distance between two Gaussians considering them as ellipsoids
	static double distEllipsoid(meanPrecisionSample &p1,meanPrecisionSample &p2);//distance between two Gaussians considering them as ellipsoids
	double muLambdaExpectation(pfn_muLambdaExpectation p,void *data);//calculates expectation of a function
	void splitGaussian(GaussianMixtureModel &GM,int id_){splitGaussian(&GM,id_);};//splits a Gaussian into two in order to test cell division
	void splitGaussian(GaussianMixtureModel *GM,int id_);//splits a Gaussian into two in order to test cell division
	void splitGaussian(GaussianMixtureModel &GM,int id_,float* img,int imSize[dimsImage]){splitGaussian(&GM,id_,img,imSize);};//splits Gaussian using k-means
	void splitGaussian(GaussianMixtureModel *GM,int id_,float* img,int imSize[dimsImage]);//splits Gaussian using k-means
	void updatePriors(double betaPercentageOfN_k,double nuPercentageOfN_k, double alphaPercentage, double alphaTotal);//update priors with previous frame state
	void updatePriorsAfterSplitProposal();
	void copyPriors();
	void killMixture(void);
	double volumeGaussian();
	void pixels2units(float scaleOrig[dimsImage]);//transform elemnts form pixels to metric units
	void units2pixels(float scaleOrig[dimsImage]);
	void updateNk(){N_k=nu_k-nu_o;};

	//Kalman filter
	void motionUpdateKalman();
	void informationUpdateKalman();
	void updateCenterWithKalmanFilter(){for(int ii=0;ii<dimsImage;ii++) {m_k(ii)=mu_KF(ii);m_o(ii)=mu_KF(ii);}};

	//operators and set/get methods
	inline bool operator< (GaussianMixtureModel const& other) const{ return(this->id<other.id);};
	GaussianMixtureModel& operator=(const GaussianMixtureModel& p);
	static void setScale(const float scale_[dimsImage]){memcpy(scale,scale_,sizeof(float)*dimsImage);};
	bool isDead(){return m_o(0)<-1e31;};
	MatrixState& getPKF(){return P_KF;};

	//input/output routines
	// write/read functions
	ostream& writeXML(ostream& os);
	static ostream& writeXMLheader(ostream &os){os<<"<?xml version=\"1.0\" encoding=\"utf-8\"?>"<<endl<<"<document>"<<endl;return os;};
	static ostream& writeXMLfooter(ostream &os){os<<"</document>"<<endl;return os;};

	//debug methods
	void writeParticlesForMatlab(string outFilename);

	//static constants
	static const int dof;//degrees of freedom in the mixtures
	static const double ellipsoidVolumeConstant;//needed to calculate ellipsoid volume
	static const double ellipseAreaConstant;//needed to calculate ellipsoid area
	static MatrixState psi;//Kalman Filter transition matrix
	static MatrixState2Obs Mobs;//Kalman Filter state to observations
	static const MatrixState Q;//Kalman Filter uncertainty in motion model

protected:


private:
	//Kalman Filter sufficient statistics
	VectorState mu_KF;//mean state vector
	MatrixState P_KF;//covariance state vector
};

bool GaussianMixtureModelPtrComp (GaussianMixtureModel* a, GaussianMixtureModel* b);

//======================================================
inline double GaussianMixtureModel::expectedLogDetCovariance(void)
{
	double out=dimsImage*log(2.0)+log(W_k.determinant());

	for(int ii=0;ii<dimsImage;ii++)
	{
		out+=gsl_sf_psi(0.5*(nu_k+1-ii));
	}

	return out;
}

//======================================================
inline double GaussianMixtureModel::expectedLogResponsivity(double totalAlpha)
{
	return gsl_sf_psi(alpha_k)-gsl_sf_psi(totalAlpha);
}

//=========================================================
inline double GaussianMixtureModel::expectedMahalanobisDistance(Matrix<double,dimsImage,1> &x_n)
{
	return (dimsImage/beta_k+nu_k*(x_n-m_k).transpose()*W_k*(x_n-m_k));
}

//============================================================
inline void GaussianMixtureModel::updateGaussianParameters(const Matrix<double,dimsImage,1> &x_k,const Matrix<double,dimsImage,dimsImage> &S_k)
{
	if(fixed==false)
	{
		beta_k=beta_o+N_k;
		m_k=(beta_o*m_o+N_k*x_k)/beta_k;
		if( W_o(dimsImage-1,dimsImage-1) < 1e-8 )//it means we have a 2D ellipsoid->just regularize W with any value (we will ignore it in the putput anyway)
		{
			W_o(dimsImage-1,dimsImage-1) = 0.5 * (W_o(0,0) + W_o(1,1));
		}
		W_k=(W_o.inverse()+N_k*S_k+(beta_o*N_k/(beta_o+N_k))*(x_k-m_o)*(x_k-m_o).transpose()).inverse();
		nu_k=nu_o+N_k;
		alpha_k=max(alpha_o+N_k,0.0);//alpha_o can be negative (improper Dirichlet prior)
	}
}
//==============================================================
//VIP:THIS FUNCTION HAS TO BE THE SAME AS GMEMupdateCUDA::GMEMregularizeWkKernel
//static const double lambdaMin=0.02;//aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0 and maxRadius=10 (adjust with scale)
//static const double lambdaMax=0.2222;//aux=scaleSigma/(minRadius*minRadius) with scaleSigma=2.0 and minRadius=3.0 (adjust with scale)  (when nuclei divide they can be very narrow)
//static const double maxExcentricity=3.0*3.0;//maximum excentricity allowed

inline void GaussianMixtureModel::regularizePrecisionMatrix(bool W4DOF)
{
	if( regularizePrecisionMatrixConstants::lambdaMax < 0.0 )
	{
		cout<<"ERROR: GaussianMixtureModel::regularizePrecisionMatrix: regularizePrecisionMatrixConstants have not been initialize"<<endl;
		exit(3);
	}

	double auxMax=regularizePrecisionMatrixConstants::lambdaMax/nu_k;//aux=scaleSigma/(minRadius*minRadius*nu_k) with scaleSigma=2.0
	double auxMin=regularizePrecisionMatrixConstants::lambdaMin/nu_k;//aux=scaleSigma/(minRadius*minRadius*nu_k) with scaleSigma=2.0

	//to adjust for scale: this values is empirical
	//If I don;t do this rescaling, I would have to find which eigenvector corresponds to Z dirction to check for min/max Radius
	//basically, maxRadius_z=scaleGMEMCUDA[0]*maxRadius_x/scaleGMEMCUDA[2]
	for(int ii=0;ii<dimsImage;ii++)
	{
		if( scale[ii] < 1e-3 )
		{
			cout<<"ERROR: GaussianMixtureModel::regularizePrecisionMatrix: scale is below zero. it shoudl be set properly before calling this function"<<endl;
			exit(3);
		}
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_k(ii,jj)/=scale[ii]*scale[jj];
		}
	}

	if ( W4DOF == true)
	{
		W_k(0,2)=0.0f;W_k(1,2)=0.0f;
		W_k(2,0)=0.0f;W_k(2,1)=0.0f;
	}

	//calculate eigenvalues and eigenvectors
	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma1(W_k);

	Matrix<double,dimsImage,1> d=sigma1.eigenvalues();
	//avoid minimum size
	if(d[0]>auxMax) d[0]=auxMax;
	if(d[1]>auxMax) d[1]=auxMax;
	if(d[2]>auxMax) d[2]=auxMax;

	//avoid maximum size
	if(d[0]<auxMin) d[0]=auxMin;
	if(d[1]<auxMin) d[1]=auxMin;
	if(d[2]<auxMin) d[2]=auxMin;

	//avoid too much excentricity
	double auxNu=d[0]/d[1];
	if(auxNu>regularizePrecisionMatrixConstants::maxExcentricity) d[0]=regularizePrecisionMatrixConstants::maxExcentricity*d[1];
	else if(1./auxNu>regularizePrecisionMatrixConstants::maxExcentricity) d[1]=regularizePrecisionMatrixConstants::maxExcentricity*d[0];

	auxNu=d[0]/d[2];
	if(auxNu>regularizePrecisionMatrixConstants::maxExcentricity) d[0]=regularizePrecisionMatrixConstants::maxExcentricity*d[2];
	else if(1./auxNu>regularizePrecisionMatrixConstants::maxExcentricity) d[2]=regularizePrecisionMatrixConstants::maxExcentricity*d[0];

	auxNu=d[1]/d[2];
	if(auxNu>regularizePrecisionMatrixConstants::maxExcentricity) d[1]=regularizePrecisionMatrixConstants::maxExcentricity*d[2];
	else if(1./auxNu>regularizePrecisionMatrixConstants::maxExcentricity) d[2]=regularizePrecisionMatrixConstants::maxExcentricity*d[1];

	//reconstruct W_k
	W_k=sigma1.eigenvectors()*d.asDiagonal()*sigma1.eigenvectors().transpose();

	//undo adjust for scale:
	//to adjust for scale: this values is empirical
	//If I don't do this rescaling, I would have to find which eigenvector corresponds to Z dirction to check for min/max Radius
	for(int ii=0;ii<dimsImage;ii++)
	{
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_k(ii,jj)*=scale[ii]*scale[jj];
		}
	}
}
//===============================================================
inline double GaussianMixtureModel::muLambdaExpectation(pfn_muLambdaExpectation p,void *data)
{
	double E=0.0;
	for(int kk=0;kk<numParticles;kk++)
	{
		E+=p(muLambdaSamples[kk],data);//weighted sum of particles
	}
	return E;
}

#endif /* GAUSSIANMIXTUREMODEL_H_ */
