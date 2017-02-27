/*
 * GammaCDF.h
 *
 *  Created on: May 20, 2011
 *      Author: amatf
 *
 *      Adapted from numerical recipes Chapter 7
 *
 *      Chi-Square(nu)=Gammadev(nu/2.0,0.5)
 */

#ifndef GAMMACDF_H_
#define GAMMACDF_H_


#include <iostream>
#include "math.h"
#include <fstream>
namespace mylib
{
	extern "C"
	{
		#include "../mylib/cdf.h"
	}
}

using namespace std;

struct Gammadev{

	double alph, oalph, bet;
	double a1,a2;
	mylib::CDF *normal;

	Gammadev(double aalph, double bbet,mylib::uint32 seed) : alph(aalph), oalph(aalph), bet(bbet)
	{

		if (alph <= 0.)
		{
			cout<<"bad alpha in Gammadev"<<endl;
			exit(2);
		}
		if (alph < 1.) alph += 1.;
		a1 = alph-1./3.;
		a2 = 1./sqrt(9.*a1);

		normal=mylib::Normal_CDF(0.0,1.0);
		mylib::Seed_CDF(normal,seed);
	};
	//destructor
	~Gammadev()
	{
		mylib::Free_CDF(normal);
	};

	//sample method
	double sample()
	{
		double u,v,x;
		do {
			do {
				x = mylib::Sample_CDF(normal);
				v = 1. + a2*x;
			}while (v <= 0.);
			v = v*v*v;
			u = mylib::drand();
		} while (u > 1. - 0.331*pow(x,4) && log(u) > 0.5*x*x + a1*(1.-v+log(v))); //Rarely evaluated.

		if (alph == oalph)
			return a1*v/bet;
		else {
			do u=mylib::drand();
			while (u == 0.);
			return pow(u,1./oalph)*a1*v/bet;
		}
	};

	//test
	void testChiSquare(string fileOut);
};

#endif /* GAMMACDF_H_ */
