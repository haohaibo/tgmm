/*
 * Ellipsoid.cpp
 *
 */

#include "AnnotationEllipsoid.h"
#include <math.h>
#include "external/xmlParser/svlStrUtils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>


//TODO: find a better random numebr generator for windows
#ifdef _WIN32 || _WIN64
	#define drand48()	(rand()*(1.0/(double)RAND_MAX))
#endif



AnnotationEllipsoid::AnnotationEllipsoid() 
{
	imgFilename = string("");
	classVal = -1;

};

AnnotationEllipsoid::AnnotationEllipsoid(const double *mu_, const double *W_, string& imgFilename_)
{
	imgFilename = imgFilename_;
	memcpy(mu,mu_, sizeof(double) * dimsImage);
	memcpy(W,W_, sizeof(double) * dimsImage * (dimsImage +1) /2);

	classVal = -1;
}

AnnotationEllipsoid::AnnotationEllipsoid(const AnnotationEllipsoid &p) 
{
	imgFilename = p.imgFilename;
	memcpy(mu,p.mu, sizeof(double) * dimsImage);
	memcpy(W,p.W, sizeof(double) * dimsImage * (dimsImage +1) /2);

	className = p.className;
	classVal = p.classVal;

	svFilename = p.svFilename;
	svIdx = p.svIdx;
}

AnnotationEllipsoid& AnnotationEllipsoid::operator=(const AnnotationEllipsoid& p)
{
	if (this != &p)
	{
		imgFilename = p.imgFilename;
		memcpy(mu,p.mu, sizeof(double) * dimsImage);
		memcpy(W,p.W, sizeof(double) * dimsImage * (dimsImage +1) /2);

		className = p.className;
		classVal = p.classVal;

		svFilename = p.svFilename;
		svIdx = p.svIdx;
	}

	return *this;
}

bool operator< (const AnnotationEllipsoid& lhs, const AnnotationEllipsoid& rhs)
{
	if( lhs.imgFilename.compare( rhs.imgFilename ) < 0 )
		return true;
	else
		return false;
}

AnnotationEllipsoid::~AnnotationEllipsoid()
{
	//nothing here yet
}

ostream& AnnotationEllipsoid::writeXML(ostream& os)
{
	os<<"<Surface ";
	os<<"name=\""<<"Ellipsoid"<<"\" id=\""<<1<<"\" numCoeffs=\""<<dimsImage * dimsImage<<"\" "<<endl;

	os<<"coeffs=\"";
	for(unsigned int ii=0;ii< dimsImage * (1+dimsImage) /2 ;ii++) os<<W[ii]<<" ";
	for(unsigned int ii=0;ii< dimsImage ;ii++) os<<mu[ii]<<" ";
	os<<"\""<<endl;

	os<<"covarianceMatrixSize=\""<<dimsImage<<"\" ";
		
	os<<" imFilename=\""<< imgFilename<<"\" class=\""<<className<<"\"";

	if( svIdx.empty() == false )
	{
		os<<" svFilename=\""<< svFilename<<"\" svIdx=\"";
		for(vector<int>::const_iterator iter = svIdx.begin(); iter != svIdx.end(); ++iter)
		{
			os<<(*iter)<<" ";
		}
		os<<"\"";
	}

	os<<"></Surface>"<<endl;

	return os;
}

//HOW TO USE readXML
//XMLNode xMainNode=XMLNode::openFileHelper("filename.xml","document");
//int n=xMainNode.nChildNode("GaussianMixtureModel");
//for(int ii=0;ii<n;ii++) GaussianMixtureModel GM(xMainNode,ii);
AnnotationEllipsoid::AnnotationEllipsoid (XMLNode &xml,int position)
{

	XMLNode node = xml.getChildNode("Surface",&position);

	XMLCSTR aux=node.getAttribute("numCoeffs");
	vector<unsigned long int> vv;
	assert(aux != NULL);
	parseString<unsigned long int>(string(aux), vv);
	unsigned int numCoeffs_=vv[0];
	vv.clear();


	aux=node.getAttribute("coeffs");
	vector<double> coeffs_;
	assert(aux != NULL);
	parseString<double>(string(aux), coeffs_);

	int ii =0;
	for(;ii<dimsImage * (1 + dimsImage) /2; ii++)
		W[ii] = coeffs_[ii];
	int jj =0;
	for(;ii < (int) numCoeffs_; ii++, jj++)
		mu[jj] = coeffs_[ii];

	aux=node.getAttribute("imFilename");
	assert(aux != NULL);
	imgFilename = string(aux);

	aux=node.getAttribute("class");
	if( aux!= NULL)
		className = string(aux);
	else
		className = "";

	classVal = -1;

	aux=node.getAttribute("svIdx");
	vector<int> dd;
	if(aux != NULL)
	{
		parseString<int>(string(aux), dd);
		svIdx = dd;
		dd.clear();
	}

	aux=node.getAttribute("svFilename");
	if(aux != NULL)
		svFilename = string(aux);


	//double check that the rest of the properties are correct
	aux=node.getAttribute("name");
	assert(aux != NULL);
	if(string(aux).compare("Ellipsoid")!=0)
	{
		cout<<"ERROR: at AnnotationEllipsoid::readXML : trying to read surface type "<<string(aux)<<" using "<<"Ellipsoid"<<" function"<<endl;
		exit(10);
	}

	aux=node.getAttribute("covarianceMatrixSize");
	assert(aux != NULL);
	parseString<unsigned long int>(string(aux), vv);
	if((int)vv[0]!=dimsImage)
	{
		cout<<"ERROR: at AnnotationEllipsoid::readXML : trying to read ellipsoid of dimensions "<<vv[0]<<" using "<<dimsImage<<" dimensions. Change covarianceMatrixSize and recompile"<<endl;
		exit(10);
	}
	vv.clear();

}

