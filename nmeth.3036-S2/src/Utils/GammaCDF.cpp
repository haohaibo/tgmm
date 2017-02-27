/*
 * GammaCDF.cpp
 *
 *  Created on: May 20, 2011
 *      Author: amatf
 */

#include "GammaCDF.h"

void Gammadev::testChiSquare(string fileOut)
{
	int numSamples=10000;

	ofstream out(fileOut.c_str());
	out<<"s1=["<<endl;
	for(int ss=0;ss<numSamples;ss++) out<<sample()<<endl;
	out<<"];"<<endl;

	//generate samples by producing Normal(0,1)
	int nu=(int)(2.0*alph);
	out<<"s2=["<<endl;

	for(int ss=0;ss<numSamples;ss++)
	{
		double aux=0.0;
		double nn;
		for(int ii=0;ii<nu;ii++)
		{
			nn=mylib::Sample_CDF(normal);
			aux+=(nn*nn);
		}
		out<<aux<<endl;
	}
	out<<"];"<<endl;

	out.close();
}
