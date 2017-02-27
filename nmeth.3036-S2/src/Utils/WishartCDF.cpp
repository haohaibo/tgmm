/*
 * WishartCDF.cpp
 *
 *  Created on: May 24, 2011
 *      Author: amatf
 */

#include "WishartCDF.h"

const double Wishartdev::Pi=3.14159265358979323846;
const double Wishartdev::constant1 = dimsImage*log(2.0);
const double Wishartdev::constant2 = 0.25*dimsImage*(dimsImage-1)*log(Wishartdev::Pi);


#if defined(_WIN32) || defined(_WIN64)
#include <stdio.h>
#include <process.h>
#define popen _popen
#define getpid _getpid
#define pclose _pclose
#endif

void Wishartdev::testWishart(string outFile)
{

	Matrix<double,dimsImage,1> mu;
	Matrix<double,dimsImage,dimsImage> sigma,lambda;
	double nu_k;

	const int numSamples=10000;
	mylib::uint32 seed=getpid();
	//give values to teh parameters
	mu<<10.3,15.2;
	sigma<<25.1,2.3,2.3,12.1;//covariance matrix

	nu_k=10.0;//it has to be larger than dimsImage
	sigma=sigma*nu_k;//so the expecte value is the covariance matrix

	lambda=sigma.inverse();


	//sample from the parameters
	Wishartdev *wishartCDF=new Wishartdev(nu_k,lambda,seed++);//bogus initialization

	cout<<"DEBUGGING: Wishart sampler "<<outFile<<endl;
	ofstream out(outFile.c_str());


	//first line is the expected value for each parameter in the proposal distribution
	for(int ii=0;ii<dimsImage;ii++) out<<mu(ii)<<" ";
	for(int ii=0;ii<dimsImage*dimsImage;ii++) out<<nu_k*lambda(ii)<<" ";
	out<<"-1.0"<<endl;

	//write out each particle
	for(int kk=0;kk<numSamples;kk++)
	{
		wishartCDF->sample(sigma);
		for(int ii=0;ii<dimsImage;ii++) out<<mu(ii)<<" ";
		for(int ii=0;ii<dimsImage*dimsImage;ii++) out<<sigma(ii)<<" ";
		out<<1.0<<endl;
	}

	out.close();
	delete wishartCDF;

}
