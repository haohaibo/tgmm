/*
 * MultinormalCDF.h
 *
 *  Created on: May 23, 2011
 *      Author: amatf
 */

#ifndef MULTINORMALCDF_H_
#define MULTINORMALCDF_H_

#include <iostream>
#include "math.h"
#include <fstream>
#include "../GaussianMixtureModel.h" //I do it for efficiency purposes, so I can define a fixed size matrix
#include "../external/Eigen/Cholesky"

namespace mylib
{
	extern "C"
	{
		#include "../mylib/cdf.h"
	}
}

using namespace std;

struct Multinormaldev{

	LLT< Matrix<double,dimsImage,dimsImage> > L_k;//cholesky decomposition of covariance matrix
	Matrix<double,dimsImage,dimsImage> lambda_k;//precision matrix

	mylib::CDF *normal;
	Matrix<double,dimsImage,1> mu;//mean of the multinormal
	double Zp,ZpLog;//normalization constant
	Matrix<double,dimsImage,1> xAux;//in case we need to evaluate using double[dimsImage]

	//statics variables
	static const double Pi;
	static const double Pi2;

	//initialization with covariance matrix
	Multinormaldev(Matrix<double,dimsImage,1> &amu,Matrix<double,dimsImage,dimsImage> &aS_k,mylib::uint32 seed)
	{

		mu=amu;
		L_k=aS_k.llt();
		normal=mylib::Normal_CDF(0.0,1.0);
		mylib::Seed_CDF(normal,seed);
		Zp=pow(Pi2,dimsImage/2.0)*L_k.matrixL().determinant();
		ZpLog=log(Zp);
		lambda_k=aS_k.inverse();//we assume dimsImage is low, so analytical formulas apply
	};
	//initialization with precision matrix
	Multinormaldev(Matrix<double,dimsImage,1> &amu,Matrix<double,dimsImage,dimsImage> &aS_kInverse,mylib::uint32 seed,bool dummy)//bool just to differentitate for above constructor
	{

		mu=amu;
		L_k=aS_kInverse.inverse().llt();//we assume dimsImage is low, so analytical formulas apply
		normal=mylib::Normal_CDF(0.0,1.0);
		mylib::Seed_CDF(normal,seed);
		Zp=pow(Pi2,dimsImage/2.0)*L_k.matrixL().determinant();
		ZpLog=log(Zp);
		lambda_k=aS_kInverse;
	};
	//destructor
	~Multinormaldev()
	{
		mylib::Free_CDF(normal);
	};
	void resetParameters(Matrix<double,dimsImage,1> &amu,Matrix<double,dimsImage,dimsImage> &aS_k)
	{
		mu=amu;
		L_k=aS_k.llt();
		Zp=pow(Pi2,dimsImage/2.0)*L_k.matrixL().determinant();
		ZpLog=log(Zp);
		lambda_k=aS_k.inverse();//we assume dimsImage is low, so analytical formulas apply
	}

	void sample(Matrix<double,dimsImage,1> &s)
	{
		for(int ii=0;ii<dimsImage;ii++) s(ii)=mylib::Sample_CDF(normal);
		//transform covariance matrix to the appropriate parameter
		s=L_k.matrixL()*s+mu;
	};

	double eval(const Matrix<double,dimsImage,1> &s)//returns the probability of a given realization
	{
		return exp(-0.5*(((s-mu).transpose()*lambda_k*(s-mu))(0)))/Zp;
	}

	double eval(double s[dimsImage])
	{
		memcpy(xAux.data(),s,sizeof(double)*dimsImage);
		return eval(xAux);
	}
	double evalLog(const Matrix<double,dimsImage,1> &s)//returns the log probability of a given realization
	{
		return -0.5*(((s-mu).transpose()*lambda_k*(s-mu))(0))-ZpLog;
	}
	double evalMahalanobisDistance(const Matrix<double,dimsImage,1> &s)//returns the log probability of a given realization
		{
			return (((s-mu).transpose()*lambda_k*(s-mu))(0));
		}
	double evalLog(double s[dimsImage])
	{
		memcpy(xAux.data(),s,sizeof(double)*dimsImage);
		return evalLog(xAux);
	}
};

#endif /* MULTINORMALCDF_H_ */
