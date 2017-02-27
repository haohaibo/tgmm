/*
 * GaussianMixtureModel.cpp
 *
 *  Created on: May 12, 2011
 *      Author: amatf
 */

#include "GaussianMixtureModel_Redux.h"
#include <float.h>
#include <vector>
#include <assert.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <cstring>
#include "external/xmlParser/svlStrUtils.h"


#if defined(_WIN32) || defined(_WIN64)
#define isnanF(x) (_isnan(x))
#else
#define isnanF(x) (isnan(x))
#endif

float GaussianMixtureModelRedux::scale[dimsImage];


GaussianMixtureModelRedux::GaussianMixtureModelRedux()
{
	resetValues();
	id=-1;
	//for(int ii=0;ii<dimsImage;ii++) scale[ii]=1.0f;
}

GaussianMixtureModelRedux::GaussianMixtureModelRedux(int id_)
{
	resetValues();
	id=id_;
	//for(int ii=0;ii<dimsImage;ii++) scale[ii]=1.0f;	
}
GaussianMixtureModelRedux::GaussianMixtureModelRedux(int id_,float scale_[dimsImage])
{
	resetValues();
	id=id_;
	for(int ii=0;ii<dimsImage;ii++) scale[ii]=scale_[ii];
	
}
//=====================================================
void GaussianMixtureModelRedux::resetValues()
{
	
	memset(m_k,0,sizeof(double)*dimsImage);
	beta_k=0.0;
	memset(W_k,0,sizeof(double)*dimsImage*dimsImage);
	nu_k=(double)dimsImage;
	alpha_k=0.0;
	N_k=0.0;
	splitScore=0.0;
	fixed=false;

	lineageId=-1;
	parentId=-1;
	color=-1;
	supervoxelIdx.clear();
}

//=============================================================
GaussianMixtureModelRedux::GaussianMixtureModelRedux(const GaussianMixtureModelRedux & p)
{

	//m_k=p.m_k;
	memcpy(m_k,p.m_k,sizeof(double)*dimsImage);
	beta_k=p.beta_k;
	//W_k=p.W_k;
	memcpy(W_k,p.W_k,sizeof(double)*dimsImage*dimsImage);
	nu_k=p.nu_k;
	alpha_k=p.alpha_k;

	//m_o=p.m_o;
	memcpy(m_o,p.m_o,sizeof(double)*dimsImage);
	beta_o=p.beta_o;
	//W_o=p.W_o;
	memcpy(W_o,p.W_o,sizeof(double)*dimsImage*dimsImage);
	nu_o=p.nu_o;
	alpha_o=p.alpha_o;

	id=p.id;
	lineageId=p.lineageId;
	parentId=p.parentId;
	N_k=p.N_k;
	splitScore=p.splitScore;
	fixed=p.fixed;
	color=p.color;



	sigmaDist_o=p.sigmaDist_o;
	supervoxelIdx = p.supervoxelIdx;

}
//===========================================================
GaussianMixtureModelRedux& GaussianMixtureModelRedux::operator=(const GaussianMixtureModelRedux& p)
{
	if (this != &p)
	{
		//m_k=p.m_k;
		memcpy(m_k,p.m_k,sizeof(double)*dimsImage);
		beta_k=p.beta_k;
		//W_k=p.W_k;
		memcpy(W_k,p.W_k,sizeof(double)*dimsImage*dimsImage);
		nu_k=p.nu_k;
		alpha_k=p.alpha_k;

		//m_o=p.m_o;
		memcpy(m_o,p.m_o,sizeof(double)*dimsImage);
		beta_o=p.beta_o;
		//W_o=p.W_o;
		memcpy(W_o,p.W_o,sizeof(double)*dimsImage*dimsImage);
		nu_o=p.nu_o;
		alpha_o=p.alpha_o;

		id=p.id;
		lineageId=p.lineageId;
		parentId=p.parentId;
		N_k=p.N_k;
		splitScore=p.splitScore;
		fixed=p.fixed;
		color=p.color;


		sigmaDist_o=p.sigmaDist_o;
		supervoxelIdx = p.supervoxelIdx;

	}
	return *this;
}
//======================================================
GaussianMixtureModelRedux::~GaussianMixtureModelRedux()
{

}


//======================================================
//HOW TO USE readXML
//XMLNode xMainNode=XMLNode::openFileHelper("filename.xml","document");
//int n=xMainNode.nChildNode("GaussianMixtureModel");
//for(int ii=0;ii<n;ii++) GaussianMixtureModel GM(xMainNode,ii);

//@warning Vector of muLambdaSamples is not recorded to avoid generating long files
GaussianMixtureModelRedux::GaussianMixtureModelRedux(XMLNode &xml,int position)
{
	N_k=0.0;


	XMLNode node = xml.getChildNode("GaussianMixtureModel",&position);

	XMLCSTR aux=node.getAttribute("id");
	vector<unsigned long long int> vv;
	assert(aux != NULL);
	parseString<unsigned long long int>(string(aux), vv);
	id=vv[0];
	vv.clear();

	aux=node.getAttribute("lineage");
	assert(aux != NULL);
	parseString<unsigned long long int>(string(aux), vv);
	lineageId=vv[0];
	vv.clear();

	aux=node.getAttribute("parent");
	assert(aux != NULL);
	parseString<unsigned long long int>(string(aux), vv);
	parentId=vv[0];
	vv.clear();

	aux=node.getAttribute("dims");
	assert(aux != NULL);
	parseString<unsigned long long int>(string(aux), vv);
	if(dimsImage!=(int)vv[0])
	{
		cout<<"ERROR: dimsImage does not agree with XML file"<<endl;
		exit(5);
	}
	vv.clear();

	aux=node.getAttribute("svIdx");
	vector<int> ll;
	if(aux != NULL)
	{
		parseString<int>(string(aux), ll);
		supervoxelIdx = ll;
		ll.clear();
	}

	aux=node.getAttribute("scale");
	vector<double> dd;
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	for(int ii=0;ii<dimsImage;ii++) scale[ii]=(float)(dd[ii]);
	dd.clear();

	aux=node.getAttribute("splitScore");
	if(aux==NULL)
		splitScore=-1e32;
	else
	{
		parseString<double>(string(aux), dd);
		splitScore=dd[0];
		dd.clear();
	}


	//variables values
	aux=node.getAttribute("nu");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	nu_k=dd[0];
	dd.clear();

	aux=node.getAttribute("beta");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	beta_k=dd[0];
	dd.clear();

	aux=node.getAttribute("alpha");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	alpha_k=dd[0];
	dd.clear();

	aux=node.getAttribute("m");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	for(int ii=0;ii<dimsImage;ii++) m_k[ii]=dd[ii];
	dd.clear();

	aux=node.getAttribute("W");
	assert(aux != NULL);
	string auxS(aux);
	std::transform(auxS.begin(), auxS.end(), auxS.begin(), ::tolower);
	std::size_t found = auxS.find("nan");
	if (found!=std::string::npos)//it contains nan elements
	{
		dd.resize( dimsImage * dimsImage );
		int count = 0;
		for(int ii=0;ii<dimsImage;ii++)
			for(int jj=0;jj<dimsImage;jj++)
			{
				if( ii == jj )
					dd[count] = 0.02;
				else 
					dd[count] = 0;
				count++;
			}
	}else{
		parseString<double>(auxS, dd);
	}
	int count=0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_k[count]=dd[count];
			count++;
		}
		dd.clear();
	

	//prior values
	aux=node.getAttribute("nuPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	nu_o=dd[0];
	dd.clear();

	aux=node.getAttribute("betaPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	beta_o=dd[0];
	dd.clear();

	aux=node.getAttribute("alphaPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	alpha_o=dd[0];
	dd.clear();

	aux=node.getAttribute("mPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	for(int ii=0;ii<dimsImage;ii++) m_o[ii]=dd[ii];
	dd.clear();

	aux=node.getAttribute("WPrior");
	assert(aux != NULL);
	auxS = string(aux);
	std::transform(auxS.begin(), auxS.end(), auxS.begin(), ::tolower);
	found = auxS.find("nan");
	if (found!=std::string::npos)//it contains nan elements
	{
		dd.resize( dimsImage * dimsImage );
		int count = 0;
		for(int ii=0;ii<dimsImage;ii++)
			for(int jj=0;jj<dimsImage;jj++)
			{
				if( ii == jj )
					dd[count] = 0.02;
				else 
					dd[count] = 0;
				count++;
			}
	}else{
		parseString<double>(auxS, dd);
	}
	
	count=0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_o[count]=dd[count];
			count++;
		}
	dd.clear();

	aux=node.getAttribute("distMRFPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	sigmaDist_o=dd[0];
	dd.clear();


}

//======================================================
ostream& GaussianMixtureModelRedux::writeXML(ostream& os)
{
	os<<"<GaussianMixtureModel ";;
	os<<"id=\""<<id<<"\" lineage=\""<<lineageId<<"\" parent=\""<<parentId<<"\" dims=\""<<dimsImage<<"\" splitScore=\""<<splitScore<<"\"";

	os<<" scale=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<scale[ii]<<" ";
	os<<"\""<<endl;

	//write variables values
	os<<"nu=\""<<nu_k<<"\" beta=\""<<beta_k<<"\" alpha=\""<<alpha_k<<"\"";
	os<<" m=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<m_k[ii]<<" ";
	os<<"\" W=\"";
	int count = 0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
		{
			os<<W_k[count++]<<" ";
		}
	os<<"\""<<endl;


	//write priors values
	os<<"nuPrior=\""<<nu_o<<"\" betaPrior=\""<<beta_o<<"\" alphaPrior=\""<<alpha_o<<"\" distMRFPrior=\""<<sigmaDist_o<<"\"";
	os<<" mPrior=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<m_o[ii]<<" ";
	os<<"\" WPrior=\"";
	count = 0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
			os<<W_o[count++]<<" ";
	
	//write supervoxel idx
	os<<"\" svIdx=\"";
	for(size_t ii = 0; ii<supervoxelIdx.size(); ii++)
		os<<supervoxelIdx[ii]<<" ";
	
	os<<"\">"<<endl;


	os<<"</GaussianMixtureModel>"<<endl;

	return os;

}


int readGaussianMixtureModelXMLfile(string filename,vector<GaussianMixtureModelRedux*> &vecGM)
{
	//delete any possible content
	for(unsigned int ii=0;ii<vecGM.size();ii++)
		delete (vecGM[ii]);
	vecGM.clear();

	//check if file exists
	ifstream fid(filename.c_str());
	if(!fid.is_open())
	{
		cout<<"ERROR: at readGaussianMixtureModelXMLfile: file "<<filename<<" could not be openned"<<endl;
		return 1;
	}
	fid.close();

	XMLNode xMainNode=XMLNode::openFileHelper(filename.c_str(),"document");
	int n=xMainNode.nChildNode("GaussianMixtureModel");
	for(int ii=0;ii<n;ii++) 
		vecGM.push_back( new GaussianMixtureModelRedux(xMainNode,ii) );

	return 0;

}
