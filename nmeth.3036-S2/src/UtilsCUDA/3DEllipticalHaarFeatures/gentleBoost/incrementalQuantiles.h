/*
 *  incrementalQuantiles.h
 *  
 *
 *  Created by Amat, Fernando on 10/13/11.
 
Inspired on numerical selection procedures
 *
 */


#ifndef INCREMENTAL_QUANTILES_H
#define INCREMENTAL_QUANTILES_H

#include <vector>
#include <math.h>
#include <algorithm>

using namespace std;

class IQagent {
	
private:
	//Object for estimating arbitrary quantile values from a continuing stream of data values. 
	static const int nbuf;	//Batch size. You may x10 if you expect > 10^6 data values. 
	int nq, nt, nd;					
	vector<double> pval,dbuf,qile; 
	double q0, qm;
	//Batch update. This function is called by add or report and should not be called directly by the user.
	void update(void);
	
public:
	IQagent() : nq(251), nt(0), nd(0), pval(nq), dbuf(nbuf), qile(nq,0.), q0(1.e99), qm(-1.e99) 
	{ //Constructor. No arguments. 251 quantiles, the rest are linearly interpolated
		for (int j=85;j<=165;j++) pval[j] = (j-75.)/100.;
		//Set general purpose array of p-values ranging from 10^-6 to 1-10^-6. You can change this if you want: 
		for (int j=84;j>=0;j--) 
		{
			pval[j] = 0.87191909*pval[j+1]; 
			pval[250-j] = 1.-pval[j];
		}
	};
	void add(double datum) {
		//Assimilate a new value from the stream.
		dbuf[nd++] = datum; 
		if (datum < q0) {q0 = datum;} 
		if (datum > qm) {qm = datum;}
		if (nd == nbuf) update();
	};
	
	double report(double p) 
	{
		//Return estimated p-quantile for the data seen so far. (E.g., p=0.5 for median.) 
		double q;
		if (nd > 0) update(); 
		int jl=0,jh=nq-1,j; 
		while (jh-jl>1) 
		{
			j = (jh+jl)>>1; 
			if (p > pval[j]) jl=j; 
			else jh=j;
		} 
		j = jl; 
		q = qile[j] + (qile[j+1]-qile[j])*(p-pval[j])/(pval[j+1]-pval[j]); 
		return max(qile[0],min(qile[nq-1],q));
	};
};

#endif INCREMENTAL_QUANTILES_H

