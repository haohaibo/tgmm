#include "incrementalQuantiles.h"

const int IQagent::nbuf=1000; //Batch size. You may x10 if you expect > 10^6 data values.

void IQagent::update(void) 
{
	//Batch updateThis function is called by add or report and should not be called directly by the user.
	int jd=0,jq=1,iq; 
	double target, told=0., tnew=0., qold, qnew; 
	vector<double> newqile(nq);	//Will be new quantiles after update. 
	sort(dbuf.begin(),dbuf.end()); 
	qold = qnew = qile[0] = newqile[0] = q0; 
	qile[nq-1] = newqile[nq-1] = qm; 
	pval[0] = min(0.5/(nt+nd),0.5*pval[1]); 
	pval[nq-1] = max(1.-0.5/(nt+nd),0.5*(1.+pval[nq-2])); 
	for (iq=1;iq<nq-1;iq++) 
	{	//Main loop over target p-values for interpolation
		target = (nt+nd)*pval[iq];	
		if (tnew < target) 
			for (;;) 
			{//Here’s the guts: We locate a succession of abscissa-ordinate pairs (qnew,tnew) that are the discontinuities of value or slope in Figure 8.5.1(c), breaking to perform an interpolation as we cross each target. 
				if (jq < nq && (jd >= nd || qile[jq] < dbuf[jd])) 
				{
					//Found slope discontinuity from old CDF.
					qnew = qile[jq]; 
					tnew = jd + nt*pval[jq++]; 
					if (tnew >= target) break;
				} else {	//Foundvaluediscontinuityfrombatchdata CDF
					qnew = dbuf[jd];	
					tnew = told; 
					if (qile[jq]>qile[jq-1]) tnew += nt*(pval[jq]-pval[jq-1])*(qnew-qold)/(qile[jq]-qile[jq-1]); 
					jd++;
					if (tnew >= target) break; 
					told = tnew++; 
					qold = qnew; 
					if (tnew >= target) break;
				} 
				told = tnew; 
				qold = qnew;
			} 
		if (tnew == told) newqile[iq] = 0.5*(qold+qnew); 
		else newqile[iq] = qold + (qnew-qold)*(target-told)/(tnew-told); 
		told = tnew; 
		qold = qnew;
	} 
	qile = newqile; 
	nt += nd; 
	nd = 0;
};
