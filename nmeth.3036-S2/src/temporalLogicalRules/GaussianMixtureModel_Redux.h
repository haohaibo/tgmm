/*
 * GaussianMixtureModel_redux.cpp
 *
 *  Created on: May 12, 2011
 *      Author: amatf
 *
 *
 *      A small subset of teh original class just to read and write xml files
 */

#ifndef GAUSSIANMIXTUREMODEL_REDUX_H_
#define GAUSSIANMIXTUREMODEL_REDUX_H_


#include "external/xmlParser/xmlParser.h"
#include <iostream>
#include "../constants.h"
#include <vector>

using namespace std;



//------------------------------------------
class GaussianMixtureModelRedux
{

public:

	int id;
	int lineageId;//to know parent (I need to add pointer to identity)
	int parentId;
	bool fixed;//fixed==true->we should not update parameters for this Gaussian mixture

	int color;//to paint reprsentations of the Gaussians

	//main variables to define a Gaussian in a mixture model
	double m_k[dimsImage];//expected value for the mean of teh Gaussian
	double beta_k;//proportionality constant to relate covariance matrix of the mean of the Gaussian and and the expected value of teh covariance itself
	double W_k[dimsImage * dimsImage];//expected value of the covariance of the Gaussian. It should be a symmetric matrix
	double nu_k;//degrees of freedom in the covariance. It has to be greater or equal than dimsImage
	double alpha_k;//responsivities. Expected value of the number of elements that belong to his Gaussian in the mixture model

	double N_k;//very important to setup priors order of magnitude
	double splitScore;//local Kullback divergence to consider splitting

	//---------------------------------------------------------------------------
	//priors
	double m_o[dimsImage];//expected value for the mean of teh Gaussian
	double beta_o;//proportionality constant to relate covariance matrix of the mean of the Gaussian and and the expected value of teh covariance itself
	double W_o[dimsImage * dimsImage];//expected value of the covariance of the Gaussian. It should be a symmetric matrix
	double nu_o;//degrees of freedom in the covariance. It has to be greater or equal than dimsImage
	double alpha_o;//responsivities. Expected value of the number of elements that belong to his Gaussian in the mixture model

	//MRF priors
	double sigmaDist_o;

	static float scale[dimsImage];//anisotropy in resolution between different dimensions

	//to store indexes of the supervoxel that the Gaussian belongs to
	vector<int> supervoxelIdx;
	//---------------------------------------------------------------------------
	//constructor/destructor
	GaussianMixtureModelRedux();
	GaussianMixtureModelRedux(int id_);
	GaussianMixtureModelRedux(int id_,float scale_[dimsImage]);
	GaussianMixtureModelRedux(XMLNode &xml,int position);
	GaussianMixtureModelRedux(const GaussianMixtureModelRedux & p);
	~GaussianMixtureModelRedux();
	void resetValues();

	//----------------------------------------
	bool isDead(){return m_o[0]<-1e31;};

	//operators
	GaussianMixtureModelRedux& operator=(const GaussianMixtureModelRedux& p);
	
	

	//input/output routines
	// write/read functions
	ostream& writeXML(ostream& os);
	static ostream& writeXMLheader(ostream &os){os<<"<?xml version=\"1.0\" encoding=\"utf-8\"?>"<<endl<<"<document>"<<endl;return os;};
	static ostream& writeXMLfooter(ostream &os){os<<"</document>"<<endl;return os;};

	
protected:


private:
	
};

int readGaussianMixtureModelXMLfile(string filename,vector<GaussianMixtureModelRedux*> &vecGM);

#endif
