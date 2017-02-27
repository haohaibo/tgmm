/*
 * WishartCDF.h
 *
 *  Created on: May 20, 2011
 *      Author: amatf
 *
 *      Based on Liu 2001 Springer-Verlag book (page 41 refering to a paper by Odell&Feiveson in 1966).
 */

#ifndef WISHARTCDF_H_
#define WISHARTCDF_H_


#include <iostream>
#include "math.h"
#include <fstream>
#include "../GaussianMixtureModel.h" //I do it for efficiency purposes, so I can define a fixed size matrix
#include "../external/Eigen/Cholesky"
#include "../external/gsl/gsl_sf_gamma.h"

namespace mylib
{
	extern "C"
	{
		#include "../mylib/cdf.h"
	}
}
#include "GammaCDF.h"

using namespace std;

struct Wishartdev{

	double nu;//degrees of freedom
	LLT< Matrix<double,dimsImage,dimsImage> > L_k;//cholesky decomposition of precision matrix
	Matrix<double,dimsImage,dimsImage> Winv;//to store inverse of parameter for eval function
	Gammadev *chiSquare[dimsImage];
	mylib::CDF *normal;
	Matrix<double,dimsImage,dimsImage> N;//temporary matrix to store normal variables
	Matrix<double,dimsImage,1> V;//temporary matrix to store chi square variables
	mylib::uint32 seed;
	double Zp,ZpLog;//normalization constant

	//static constants
	static const double Pi;
	static const double constant1;
	static const double constant2;

	//constructor
	Wishartdev(double anu,Matrix<double,dimsImage,dimsImage> &aW_k,mylib::uint32 aseed) : nu(anu)
	{
		L_k=aW_k.llt();
		normal=mylib::Normal_CDF(0.0,1.0);
		seed=aseed;
		mylib::Seed_CDF(normal,seed++);
		for(int ii=0;ii<dimsImage;ii++) chiSquare[ii]=new Gammadev((nu-ii)/2.0,0.5,seed++);

		Winv=aW_k.inverse();//we assume dimsImage<5 so we can use analytical formulas for inverse
		//calculate constants
		double aux=0.5*nu;
		ZpLog=aux*log(aW_k.determinant())+aux*constant1+constant2;
		for(int ii=0;ii<dimsImage;ii++) ZpLog+=gsl_sf_lngamma((nu-ii)/2.0);
		Zp=exp(ZpLog);
	};
	//destructor
	~Wishartdev()
	{
		for(int ii=0;ii<dimsImage;ii++) delete chiSquare[ii];
		mylib::Free_CDF(normal);
	};
	//reset parameters
	void resetParameters(double anu,Matrix<double,dimsImage,dimsImage> &aW_k)
	{
		L_k=aW_k.llt();
		nu=anu;
		for(int ii=0;ii<dimsImage;ii++)
		{
			delete chiSquare[ii];
			chiSquare[ii]=new Gammadev((nu-ii)/2.0,0.5,seed++);
		}

		Winv=aW_k.inverse();//we assume dimsImage<5 so we can use analytical formulas for inverse

		//recalculate partition function value
		double aux=0.5*nu;
		ZpLog=aux*log(aW_k.determinant())+aux*constant1+constant2;
		for(int ii=0;ii<dimsImage;ii++) ZpLog+=gsl_sf_lngamma((nu-ii)/2.0);
		Zp=exp(ZpLog);
	}
	//sample method TODO Write (hard coded) special case for low dimensions to speed up code and avoid so many for loops (manual loop unrolling)
	void sample(Matrix<double,dimsImage,dimsImage> &W)
	{
		//generate normal and chi-square samples
		for(int ii=0;ii<dimsImage;ii++)
		{
			V(ii)=chiSquare[ii]->sample();
			for(int jj=ii+1;jj<dimsImage;jj++)
			{
				N(ii,jj)=mylib::Sample_CDF(normal);
			}
		}

		//compute covariance matrix
		for(int ii=0;ii<dimsImage;ii++)
		{
			W(ii,ii)=V(ii);
			for(int rr=0;rr<ii;rr++)
				W(ii,ii)+=N(rr,ii)*N(rr,ii);

			for(int jj=ii+1;jj<dimsImage;jj++)
			{
				W(ii,jj)=N(ii,jj)*sqrt(V(ii));
				for(int rr=0;rr<ii;rr++)
					W(ii,ii)+=N(rr,ii)*N(rr,jj);
				//symmetric
				W(jj,ii)=W(ii,jj);
			}
		}

		//transform covariance matrix to the appropriate parameter
		W=L_k.matrixL()*W*L_k.matrixL().transpose();
	};

	double eval(const Matrix<double,dimsImage,dimsImage> &W)
	{
		return(pow(W.determinant(),(nu-dimsImage-1.0)/2.0)*exp(-0.5*(Winv*W).trace())/Zp);
	}
	double evalLog(const Matrix<double,dimsImage,dimsImage> &W)
	{
		return (log(W.determinant())*((nu-dimsImage-1.0)/2.0)-0.5*(Winv*W).trace()-ZpLog);
	}

	//debug mode
	static void testWishart(string outFile);
};

#endif /* WISHARTCDF_H_ */
