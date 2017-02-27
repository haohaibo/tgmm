/*
 * GaussianMixtureModel.cpp
 *
 *  Created on: May 12, 2011
 *      Author: amatf
 */

#include "GaussianMixtureModel.h"
#include <float.h>
#include <vector>
#include <cmath>
#include "external/xmlParser2/svlStrUtils.h"
#include "Utils/WishartCDF.h"
#include "Utils/MultinormalCDF.h"


#if defined(_WIN32) || defined(_WIN64)
#define isnanF(x) (_isnan(x))
#else
#define isnanF(x) (std::isnan(x))
#endif

float GaussianMixtureModel::scale[dimsImage];
const int GaussianMixtureModel::dof=dimsImage+dimsImage*(dimsImage+1)/2;
const double GaussianMixtureModel::ellipsoidVolumeConstant=(4.0/3.0)*3.14159265358979310*pow(2.0,dimsImage);//we return the volume of an ellipsoid of radius 2*sigma in each dimension
const double GaussianMixtureModel::ellipseAreaConstant=3.14159265358979310*pow(2.0,dimsImage);//we return the volume of an ellipsoid of radius 2*sigma in each dimension

MatrixState GaussianMixtureModel::psi=MatrixState::Identity();//Kalman Filter transition matrix
MatrixState2Obs GaussianMixtureModel::Mobs=MatrixState2Obs::Zero();//Kalman Filter state to observations
const MatrixState GaussianMixtureModel::Q=MatrixState::Identity();//Kalman Filter uncertainty in motion model


const double meanPrecisionSample::nuProposal = 10.0;
const double meanPrecisionSample::betaProposal = 3.0*3.0;//if we were in one dimension sigma=sigmaEllipsoid/sqrt(betaProposal). So with 3.0 uncertainty of mean is all inside the cell size


GaussianMixtureModel::GaussianMixtureModel()
{
	resetValues();
	id=-1;
	supervoxelIdx.reserve(3);
	//for(int ii=0;ii<dimsImage;ii++) scale[ii]=1.0f;
}

GaussianMixtureModel::GaussianMixtureModel(int id_)
{
	resetValues();
	id=id_;
	supervoxelIdx.reserve(3);
	//for(int ii=0;ii<dimsImage;ii++) scale[ii]=1.0f;
}
GaussianMixtureModel::GaussianMixtureModel(int id_,float scale_[dimsImage])
{
	resetValues();
	id=id_;
	supervoxelIdx.reserve(3);
	for(int ii=0;ii<dimsImage;ii++) scale[ii]=scale_[ii];
}
//=====================================================
void GaussianMixtureModel::resetValues()
{
	m_k=Matrix<double,dimsImage,1>::Zero();
	beta_k=0.0;
	W_k=Matrix<double,dimsImage,dimsImage>::Zero();
	nu_k=(double)dimsImage;
	alpha_k=0.0;
	N_k=0.0;
	splitScore=0.0;
	fixed=false;

	lineageId=-1;
	parentId=-1;
	color=-1;

	//Kalman filter parameters
	for(int ii=0;ii<dimsImage;ii++) psi(ii,ii+dimsImage)=1.0;
	for(int ii=0;ii<dimsImage;ii++) Mobs(ii,ii)=1.0;
	mu_KF=VectorState::Zero();
	P_KF=MatrixState::Identity();
	supervoxelIdx.clear();
}

//=============================================================
GaussianMixtureModel::GaussianMixtureModel(const GaussianMixtureModel & p)
{

	m_k=p.m_k;
	beta_k=p.beta_k;
	W_k=p.W_k;
	nu_k=p.nu_k;
	alpha_k=p.alpha_k;

	m_o=p.m_o;
	beta_o=p.beta_o;
	W_o=p.W_o;
	nu_o=p.nu_o;
	alpha_o=p.alpha_o;

	id=p.id;
	lineageId=p.lineageId;
	parentId=p.parentId;
	N_k=p.N_k;
	splitScore=p.splitScore;
	fixed=p.fixed;
	color=p.color;

	dist=p.dist;
	memcpy(center,p.center,sizeof(float)*dimsImage);

	memcpy(muLambdaSamples,p.muLambdaSamples,sizeof(meanPrecisionSample)*numParticles);
	sigmaDist_o=p.sigmaDist_o;

	mu_KF=p.mu_KF;
	P_KF=p.P_KF;

	supervoxelIdx = p.supervoxelIdx;
}
//===========================================================
GaussianMixtureModel& GaussianMixtureModel::operator=(const GaussianMixtureModel& p)
{
	if (this != &p)
	{
		m_k=p.m_k;
		beta_k=p.beta_k;
		W_k=p.W_k;
		nu_k=p.nu_k;
		alpha_k=p.alpha_k;

		m_o=p.m_o;
		beta_o=p.beta_o;
		W_o=p.W_o;
		nu_o=p.nu_o;
		alpha_o=p.alpha_o;

		id=p.id;
		lineageId=p.lineageId;
		parentId=p.parentId;
		N_k=p.N_k;
		splitScore=p.splitScore;
		fixed=p.fixed;
		color=p.color;

		dist=p.dist;
		memcpy(center,p.center,sizeof(float)*dimsImage);

		memcpy(muLambdaSamples,p.muLambdaSamples,sizeof(meanPrecisionSample)*numParticles);
		sigmaDist_o=p.sigmaDist_o;

		mu_KF=p.mu_KF;
		P_KF=p.P_KF;

		supervoxelIdx = p.supervoxelIdx;
	}
	return *this;
}
//======================================================
GaussianMixtureModel::~GaussianMixtureModel()
{

}

//=====================================================
float GaussianMixtureModel::distBlobs(const float *cc)
{
	float dd=0;
	for(int ii=0;ii<dimsImage;ii++) dd+=scale[ii]*scale[ii]*(center[ii]-cc[ii])*(center[ii]-cc[ii]);

	return sqrt(dd);
}
//======================================================
//HOW TO USE readXML
//XMLNode xMainNode=XMLNode::openFileHelper("filename.xml","document");
//int n=xMainNode.nChildNode("GaussianMixtureModel");
//for(int ii=0;ii<n;ii++) GaussianMixtureModel GM(xMainNode,ii);

//@warning Vector of muLambdaSamples is not recorded to avoid generating long files
GaussianMixtureModel::GaussianMixtureModel(XMLNode &xml,int position)
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
	for(int ii=0;ii<dimsImage;ii++) m_k(ii)=dd[ii];
	dd.clear();

	aux=node.getAttribute("W");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	int count=0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_k(ii,jj)=dd[count];
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
	for(int ii=0;ii<dimsImage;ii++) m_o(ii)=dd[ii];
	dd.clear();

	aux=node.getAttribute("WPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	count=0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_o(ii,jj)=dd[count];
			count++;
		}
	dd.clear();

	aux=node.getAttribute("distMRFPrior");
	assert(aux != NULL);
	parseString<double>(string(aux), dd);
	sigmaDist_o=dd[0];
	dd.clear();

	//Kalman Filter parameters
	for(int ii=0;ii<dimsImage;ii++) psi(ii,ii+dimsImage)=1.0;
	for(int ii=0;ii<dimsImage;ii++) Mobs(ii,ii)=1.0;
	mu_KF=VectorState::Zero();//we assume no speed for initial frame
	for(int ii=0;ii<dimsImage;ii++) mu_KF(ii)=m_k(ii);
	P_KF=MatrixState::Identity();
}

//======================================================
ostream& GaussianMixtureModel::writeXML(ostream& os)
{
	os<<"<GaussianMixtureModel ";;
	os<<"id=\""<<id<<"\" lineage=\""<<lineageId<<"\" parent=\""<<parentId<<"\" dims=\""<<dimsImage<<"\" splitScore=\""<<splitScore<<"\"";

	os<<" scale=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<scale[ii]<<" ";
	os<<"\""<<endl;

	//write variables values
	os<<"nu=\""<<nu_k<<"\" beta=\""<<beta_k<<"\" alpha=\""<<alpha_k<<"\"";
	os<<" m=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<m_k(ii)<<" ";
	os<<"\" W=\"";
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
			os<<W_k(ii,jj)<<" ";
	os<<"\""<<endl;

	//write priors values
	os<<"nuPrior=\""<<nu_o<<"\" betaPrior=\""<<beta_o<<"\" alphaPrior=\""<<alpha_o<<"\" distMRFPrior=\""<<sigmaDist_o<<"\"";
	os<<" mPrior=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<m_o(ii)<<" ";
	os<<"\" WPrior=\"";
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=0;jj<dimsImage;jj++)
			os<<W_o(ii,jj)<<" ";
	
	//write supervoxel idx
	os<<"\" svIdx=\"";
	for(size_t ii = 0; ii<supervoxelIdx.size(); ii++)
		os<<supervoxelIdx[ii]<<" ";
	
	os<<"\">"<<endl;


	os<<"</GaussianMixtureModel>"<<endl;

	return os;

}

//=============================================================
bool GaussianMixtureModelPtrComp (GaussianMixtureModel* a, GaussianMixtureModel* b)
{
	return(a->id<b->id);
}

//=============================================================
//==============================================================================================================================================
double GaussianMixtureModel::distEllipsoid(GaussianMixtureModel &p)
{
	meanPrecisionSample p1;
	p1.lambda_k=W_k*nu_k;
	p1.mu_k=m_k;

	meanPrecisionSample p2;
	p2.lambda_k=p.W_k*p.nu_k;
	p2.mu_k=p.m_k;
	return distEllipsoid(p1,p2);
}
//===================================================================
double GaussianMixtureModel::distEllipsoid(meanPrecisionSample &p1,meanPrecisionSample &p2)
{
	if(&p1==&p2){return 0.0;};


	//first quick check based on largest radius

	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma1eig(p1.lambda_k);
	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma2eig(p2.lambda_k);

	double gamma1=1.0/(sigma1eig.eigenvalues().maxCoeff());
	double gamma2=1.0/(sigma2eig.eigenvalues().maxCoeff());

	//double maxRadius1=sqrt(gamma1);//sqrtinverse of the spectral radius)
	//double maxRadius2=sqrt(gamma2);
	//double dd=(p1.mu_k-p2.mu_k).norm();
	//if(dd>(maxRadius1+maxRadius2)){*dist=1e32; return false;};

	if((p1.mu_k.transpose()*p2.lambda_k*p1.mu_k)(0)<=1.0){return 0.0;};
	if((p2.mu_k.transpose()*p1.lambda_k*p2.mu_k)(0)<=1.0){return 0.0;};

	//longer check based on algorithm described in
	//	[1] A. Lin and S. Han, ÒOn the Distance between Two Ellipsoids,Ó SIAM Journal on Optimization, vol. 13, pp. 298Ð308, May. 2002.

	Matrix<double,dimsImage,1> g1=p1.mu_k;
	Matrix<double,dimsImage,1> g2=p2.mu_k;//centers fo ball within the sphere


	//variables needed for the iterative algorithm
	double t1=0.0,t2=0.0;
	Matrix<double,dimsImage,1> a;//auxiliar to solve quadratic equation
	Matrix<double,dimsImage,1> b;//auxiliar to solve quadratic equatio
	Matrix<double,dimsImage,1> x1;//point on the surface of the ellipsoid
	Matrix<double,dimsImage,1> x2;
	double H,I,D;
	double theta1,theta2;
	//find intersections points of the line between g1,g2
	const int maxIter=50;//it converges really fast
	int numIter=0;
	while(numIter<maxIter)
	{
		//t1
		a=g2-g1;
		b=g1-p1.mu_k;
		H=a.transpose()*p1.lambda_k*a;
		I=2*a.transpose()*p1.lambda_k*b;
		D=(b.transpose()*p1.lambda_k*b)-1;
		t1=(-I+sqrt(I*I-4*H*D))/(2*H);
		x1=g1+t1*a;
		//t2
		b=g1-p2.mu_k;
		H=a.transpose()*p2.lambda_k*a;
		I=2*a.transpose()*p2.lambda_k*b;
		D=(b.transpose()*p2.lambda_k*b)-1;
		t2=(-I-sqrt(I*I-4*H*D))/(2*H);
		if(t1>=t2){return 0.0;};//ellipsoids intersect
		x2=g1+t2*a;

		//calculate new centers
		g1=p1.lambda_k*(x1-p1.mu_k);
		theta1=acos(((x2-x1).normalized()).dot(g1.normalized()));
		if( isnanF(theta1) == true ) theta1=0.0;//due to numarical imprecision when dot product is very close to 1.0
		g2=p2.lambda_k*(x2-p2.mu_k);
		theta2=acos(((x1-x2).normalized()).dot(g2.normalized()));
		if(isnanF(theta2) == true ) theta2=0.0;
		if(fabs(theta1-theta2)<1e-3)
		{
			return (x2-x1).norm();
		}//algorithm has converged and ellipsoid do not intersect

		g1*=(-gamma1);
		g1+=x1;
		g2*=(-gamma2);
		g2+=x2;
		numIter++;

		//cout<<numIter<<" "<<theta1<<" "<<theta2<<" "<<fabs(theta1-theta2)<<" "<<(x2-x1).norm()<<endl;
	}

	cout<<"ERROR: at GaussianMixtureModel::distEllipsoid reached maximum number iterations wihtout convergence"<<endl;
	cout<<"First ellipsoid:"<<endl;
	cout<<p1.mu_k<<endl;
	cout<<p1.lambda_k<<endl;
	cout<<"Second ellipsoid:"<<endl;
		cout<<p2.mu_k<<endl;
		cout<<p2.lambda_k<<endl;
		cout<<"Current distance "<<(x2-x1).norm()<<endl;

		exit(2);
	return false;//just so compiler does not complain

}

//====================================================================
void GaussianMixtureModel::writeParticlesForMatlab(string outFilename)
{
	cout<<"DEBUGGING: GaussianMixtureModel::writeParticlesForMatlab in "<<outFilename<<endl;
	ofstream out(outFilename.c_str());

	//first line is the expected value for each parameter in the proposal distribution
	for(int ii=0;ii<dimsImage;ii++) out<<m_k(ii)<<" ";
	for(int ii=0;ii<dimsImage*dimsImage;ii++) out<<nu_k*W_k(ii)<<" ";
	out<<"-1.0"<<endl;

	//write out each particle
	for(int kk=0;kk<numParticles;kk++)
	{
		for(int ii=0;ii<dimsImage;ii++) out<<muLambdaSamples[kk].mu_k(ii)<<" ";
		for(int ii=0;ii<dimsImage*dimsImage;ii++) out<<muLambdaSamples[kk].lambda_k(ii)<<" ";
		out<<muLambdaSamples[kk].w<<endl;
	}

	out.close();
}

 //using longer axis to decide mu+-sigma for the new two centers
void GaussianMixtureModel::splitGaussian(GaussianMixtureModel *GM,int id_)
{
	//compute eigenvalue decomposition
	Matrix<double,dimsImage,dimsImage> sigma1=W_k*nu_k;
	//we need to compensate for scale
	for(int ii=0;ii<dimsImage;ii++)
	{
		for(int jj=0;jj<dimsImage;jj++)
		{
			sigma1(ii,jj)/=(scale[ii]*scale[jj]);
		}
	}
	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma1eig(sigma1);

	int pos;
	double sigmaMax=sigma1eig.eigenvalues().minCoeff(&pos);
	if(sigmaMax<1e-3) sigmaMax=1e-3;//you don't want to shoot too far
	double sigmaMax2=1.0/sqrt(sigmaMax);


	Matrix<double,dimsImage,1> v=sigma1eig.eigenvectors().col(pos);//maximum change direction
	double sigmaMin=sigma1eig.eigenvalues().maxCoeff();//I can use the max coeff because I have previously adjusted for scale

	//reset values
	W_k=(sigmaMin/nu_k)*Matrix<double,dimsImage,dimsImage>::Identity();//we want to generate larger Gaussians because it is easier to compress than to expand
	//we need to compensate for scale
	for(int ii=0;ii<dimsImage;ii++) W_k(ii,ii)*=(scale[ii]*scale[ii]);
	//regularize W (just in case): not needed because the new W_k is guaranteed to pass all the tests


	m_k+=sigmaMax2*v;
	N_k*=0.5;

	//copy values to new blob
	(*GM)=(*this);
	GM->m_k-=2.0*sigmaMax2*v;
	GM->id=id_;
}

//using kmeans to decide how to split the Gaussian
void GaussianMixtureModel::splitGaussian(GaussianMixtureModel *GM,int id_,float* img,int imSize[dimsImage])
{

	int maxIter=100;
	double minDistLocalMaxima2=6.0*6.0;//squared to avoid computing so many sqrt
	double scaleSigma=4.0;//how many sigmas away we use to setup the box
	int maxWinSize=25,minWinSize=3;

	Matrix<double,dimsImage,1> auxC1,auxC2,auxCol,mkOrig=m_k;
	double aux;

	//compute eigenvalue decomposition
	Matrix<double,dimsImage,dimsImage> sigma1=W_k*nu_k;

	//find size of the box
	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma1eig(sigma1);
	int xMax=0,xMin=imSize[0]-1,yMax=0,yMin=imSize[1]-1,zMax=0,zMin=imSize[2]-1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		aux=sigma1eig.eigenvalues()(ii);
		aux=scaleSigma*1./sqrt(max(aux,1e-2));
		auxC1=aux*sigma1eig.eigenvectors().col(ii).cwiseAbs();
		xMin=min(xMin,(int)(m_k(0)-auxC1(0)));
		yMin=min(yMin,(int)(m_k(1)-auxC1(1)));
		zMin=min(zMin,(int)(m_k(2)-auxC1(2)));

		xMax=max(xMax,(int)(m_k(0)+auxC1(0)));
		yMax=max(yMax,(int)(m_k(1)+auxC1(1)));
		zMax=max(zMax,(int)(m_k(2)+auxC1(2)));
	}
	//make sure box dimensions are Ok
	if(xMax-xMin>2*maxWinSize)
	{
		xMax=(int)(m_k(0))+maxWinSize;
		xMin=(int)(m_k(0))-maxWinSize;
	}
	if(yMax-yMin>2*maxWinSize)
	{
		yMax=(int)(m_k(1))+maxWinSize;
		yMin=(int)(m_k(1))-maxWinSize;
	}
	if(zMax-zMin>2*maxWinSize)
	{
		zMax=(int)(m_k(2))+maxWinSize;
		zMin=(int)(m_k(2))-maxWinSize;
	}
	if(xMax-xMin<2*minWinSize)
	{
		xMax=(int)(m_k(0))+minWinSize;
		xMin=(int)(m_k(0))-minWinSize;
	}
	if(yMax-yMin<2*minWinSize)
	{
		yMax=(int)(m_k(1))+minWinSize;
		yMin=(int)(m_k(1))-minWinSize;
	}
	if(zMax-zMin<2*minWinSize)
	{
		zMax=(int)(m_k(2))+minWinSize;
		zMin=(int)(m_k(2))-minWinSize;
	}
	xMin=max((int)0,xMin);yMin=max((int)0,yMin);zMin=max((int)0,zMin);
	xMax=min(imSize[0]-1,xMax);yMax=min(imSize[1]-1,yMax);zMax=min(imSize[2]-1,zMax);

	long long int numElemBox=(xMax-xMin+1)*(yMax-yMin+1)*(zMax-zMin+1);

	//build weights
	Matrix<double,Dynamic,Dynamic> xn(dimsImage,numElemBox);
	VectorXd wn(numElemBox);
	//order from mylib::Array (... (xn-1*dims[n-2] + xn-2)*dims[n-3] + ...)*dims[0] + x0. For mylib dims[0] is x
	long long int pos=0;
	long long int count=-1;

	sigma1*=(-0.5);//precompute Gaussians constants on exp.//we do not need for it to decay slower: it avoids the outsiders influence

	auxC1=-1e32*Matrix<double,dimsImage,1>::Ones();
	auxC2=-1e32*Matrix<double,dimsImage,1>::Ones();//to help initialization
	double auxB1=0,auxB2=0;
	Matrix<double,dimsImage,1> auxScale;
	for(int ii=0;ii<dimsImage;ii++) auxScale(ii)=scale[ii];
	for(int zz=zMin;zz<=zMax;zz++)
	{
		for(int yy=yMin;yy<=yMax;yy++)
		{
			pos=(zz*imSize[1]+yy)*imSize[0]+xMin;
			for(int xx=xMin;xx<=xMax;xx++)
			{
				auxCol(0)=xx;auxCol(1)=yy;auxCol(2)=zz;
				aux=(auxCol-m_k).transpose()*sigma1*(auxCol-m_k);
				if(aux<-25) aux=1.3888e-11;//exp(-25)=1.3888e-11
				else aux=img[pos]*(exp(aux));
				if(aux<0)
				{
					cout<<"ERROR at GaussianMixtureModel::splitGaussian: weight can not be negative!"<<endl;
					exit(3);
				}

				if(aux<1e-8)
				{
					pos++;
					continue;//to ignore points with very low weight
				}

				//update local minima
				if(aux>auxB1 || aux>auxB2)
				{
					//make sure new maximum is far enough
					if((auxCol-auxC1).cwiseProduct(auxScale).squaredNorm()<minDistLocalMaxima2)//very close to C1
					{
						if(aux>auxB1)
						{
							auxB1=aux;
							auxC1=auxCol;
						}
					}else if((auxCol-auxC2).cwiseProduct(auxScale).squaredNorm()<minDistLocalMaxima2){//very close to C2
						if(aux>auxB2)
						{
							auxB2=aux;
							auxC2=auxCol;
						}
					}else{//not close to anything
						if(auxB1>auxB2)//replace B2
						{
							if(aux>auxB2)
							{
								auxB2=aux;
								auxC2=auxCol;
							}
						}else{//replace auxB1
							if(aux>auxB1)
							{
								auxB1=aux;
								auxC1=auxCol;
							}
						}
					}
				}
				count++;//otherwise the last value might be bogus
				xn.col(count)=auxCol;
				wn(count)=aux;
				pos++;
			}
		}
	}

	//delete unnecessary memory: not really needed since we access them using for loops( wejust restrict teh counter)
	numElemBox=count+1;
	//wn.conservativeResize(numElemBox);
	//xn.conservativeResize(dimsImage,numElemBox);

	//copy values to new blob so I can update values
	N_k*=0.5;
	(*GM)=(*this);
	GM->id=id_;


	//run k-means

	//initialize centroids by local maxima in intensity
	GM->m_k=auxC2;
	//auxC2=m_k;//auxC2 contains the original m_k: just in case we want to verify proposal
	m_k=auxC1;

	double totalW1=0.0,totalW2=0.0,totalDist=0.0,totalDistOld=0.0;
	totalDist=1e300;//guaranteed to be above the new dist
	int iter=0;
	double d1,d2;
	double *wnPtr=wn.data();

	do{
		totalDistOld=totalDist;

		//calculate new centroids
		auxC1=Matrix<double,dimsImage,1>::Zero();totalW1=0.0;
		auxC2=Matrix<double,dimsImage,1>::Zero();totalW2=0.0;
		totalDist=0.0;
		for(long long int ii=0;ii<numElemBox;ii++)
		{
			auxCol=xn.col(ii);
			d1=(auxCol-m_k).squaredNorm();
			d2=(auxCol-GM->m_k).squaredNorm();
			if(d1<d2)//assign to d1
			{
				totalW1+=wnPtr[ii];
				auxC1+=wnPtr[ii]*auxCol;
				totalDist+=d1;
			}else{//assign to d2
				totalW2+=wnPtr[ii];
				auxC2+=wnPtr[ii]*auxCol;
				totalDist+=d2;
			}

		}

		if(totalW1<1e-8)//empty cluster
		{
			m_k=-1e32*Matrix<double,dimsImage,1>::Ones();//dead cell
			totalDist=-1.0;//guaranteed to finish
		}else{
			m_k=auxC1/totalW1;
		}
		if(totalW2<1e-8)//empty cluster
		{
			GM->m_k=-1e32*Matrix<double,dimsImage,1>::Ones();//dead cell
			totalDist=-1.0;//guaranteed to finish
		}else{
			GM->m_k=auxC2/totalW2;
		}

		//cerr<<"Iter="<<iter<<";totalW1="<<totalW1<<";totalW2="<<totalW2<<";totalDist="<<totalDist<<";m_kOrig="<<mkOrig.transpose()<<";m1="<<m_k.transpose()<<";m2="<<GM->m_k.transpose()<<endl;

		iter++;
	}while(iter<maxIter && totalDist<totalDistOld);

	//calculate precision matrix for each label
	W_k=Matrix<double,dimsImage,dimsImage>::Zero();totalW1=0.0;
	GM->W_k=Matrix<double,dimsImage,dimsImage>::Zero();totalW2=0.0;
	for(long long int ii=0;ii<numElemBox;ii++)
	{
		auxCol=xn.col(ii);
		d1=(auxCol-m_k).squaredNorm();
		d2=(auxCol-GM->m_k).squaredNorm();
		if(d1<d2)//assign to d1
		{
			totalW1+=wnPtr[ii];
			W_k+=wnPtr[ii]*(auxCol-m_k)*(auxCol-m_k).transpose();
		}else{//assign to d2
			totalW2+=wnPtr[ii];
			GM->W_k+=wnPtr[ii]*(auxCol-GM->m_k)*(auxCol-GM->m_k).transpose();
		}
	}
	if(totalW1<1e-8)//empty cluster
	{
		m_k=-1e32*Matrix<double,dimsImage,1>::Ones();//dead cell
		W_k=Matrix<double,dimsImage,dimsImage>::Identity();
	}else{
		W_k/=totalW1;

		if( W_k(dimsImage-1,dimsImage-1) < 1e-8 )//it means we have a 2D ellipsoid->just regularize W with any value (we will ignore it in the putput anyway)
		{
			W_k(dimsImage-1,dimsImage-1) = 0.5 * (W_k(0,0) + W_k(1,1));
		}
		W_k=W_k.inverse()/nu_k;
	}
	if(totalW2<1e-8)//empty cluster
	{
		GM->m_k=-1e32*Matrix<double,dimsImage,1>::Ones();//dead cell
		GM->W_k=Matrix<double,dimsImage,dimsImage>::Identity();
	}else{
		GM->W_k/=totalW2;
		if( GM->W_k(dimsImage-1,dimsImage-1) < 1e-8 )//it means we have a 2D ellipsoid->just regularize W with any value (we will ignore it in the putput anyway)
		{
			GM->W_k(dimsImage-1,dimsImage-1) = 0.5 * (GM->W_k(0,0) + GM->W_k(1,1));
		}
		GM->W_k=GM->W_k.inverse()/GM->nu_k;
	}


	if(isnanF(GM->W_k(0)))
	{
		cout<<"WARNING at GaussianMixtureModel::splitGaussian: precision matrix GM->W_k is nan"<<endl;
		cout<<"Id="<<id<<" "<<id_<<";NumIters="<<iter<<";totalW1="<<totalW1<<";totalW2="<<totalW2<<";totalDist="<<totalDist<<";m_kOrig="<<mkOrig.transpose()<<";m1="<<m_k.transpose()<<";m2="<<GM->m_k.transpose()<<";nu_k="<<GM->nu_k<<endl;
		//exit(4);
		//just set some value to keep working (from regularize precision matrix)
		GM->W_k = Matrix<double,dimsImage,dimsImage>::Identity();
		for(int ii = 0; ii < dimsImage; ii++)
			GM->W_k(ii,ii) = regularizePrecisionMatrixConstants::lambdaMax * scale[ii] * scale[ii];
	}
	if( isnanF(W_k(0)) )
	{
		cout<<"WARNING at GaussianMixtureModel::splitGaussian: precision matrix W_k is nan"<<endl;
		cout<<"Id="<<id<<" "<<id_<<";NumIters="<<iter<<";totalW1="<<totalW1<<";totalW2="<<totalW2<<";totalDist="<<totalDist<<";m_kOrig="<<mkOrig.transpose()<<";m1="<<m_k.transpose()<<";m2="<<GM->m_k.transpose()<<";nu_k="<<GM->nu_k<<endl;
		//exit(4);
		//just set some value to keep working (from regularize precision matrix)
		W_k = Matrix<double,dimsImage,dimsImage>::Identity();
		for(int ii = 0; ii < dimsImage; ii++)
			W_k(ii,ii) = regularizePrecisionMatrixConstants::lambdaMax * scale[ii] * scale[ii];
	}


	//regularize precision matrix
	regularizePrecisionMatrix(true);
	GM->regularizePrecisionMatrix(true);
}

/*
//generating two Gaussians than encompass the old Gaussian
void GaussianMixtureModel::splitGaussian(GaussianMixtureModel *GM,int id_)
{
	cout<<"Splitting Gaussians with big blobs!!"<<endl;
	//compute eigenvalue decomposition
	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma1eig(W_k*nu_k);

	int pos;
	double sigmaMax=sigma1eig.eigenvalues().minCoeff(&pos);
	Matrix<double,dimsImage,1> v=sigma1eig.eigenvectors().col(pos);//maximum change direction

	//reset values
	W_k=(sigmaMax*0.8/nu_k)*Matrix<double,dimsImage,dimsImage>::Identity();
	m_k+=v;
	N_k*=0.5;

	//copy values to new blob
	(*GM)=(*this);
	GM->m_k-=2.0*v;
	GM->id=id_;
}
*/

//==============================================================================
void GaussianMixtureModel::updatePriors(double betaPercentageOfN_k,double nuPercentageOfN_k,double alphaPercentage, double alphaTotal)
{
	m_o=m_k;
	beta_o=max(betaPercentageOfN_k*N_k,1e-5);//it can never be zero
	nu_o=max(nuPercentageOfN_k*N_k,(double)dimsImage);//degrees of freedom have to be greater or equal to dimsImage
	W_o=W_k*nu_k/nu_o;
	if(alphaTotal<0.0)
		alpha_o=alpha_o;//this value does not need to be altered
	else
		alpha_o=alphaTotal*minPi_kForDead*alphaPercentage;//we only kill very poor mixtures. Multiplier has to be<1.0 , otherwise even with N_k=0 Gsuassians will still survive
}
//==============================================================================
void GaussianMixtureModel::updatePriorsAfterSplitProposal()
{
	m_o=m_k;
	beta_o=max(0.5*N_k,1e-5);//it can never be zero
	nu_o=max(0.5*N_k,(double)dimsImage);//degrees of freedom have to be greater or equal to dimsImage
	W_o=W_k*nu_k/nu_o;

	//do not touch alpha:it should be OK
	//alpha_o=alphaTotal*minPi_kForDead*0.8;//we only kill very poor mixtures. Multiplier has to be<1.0 , otherwise even with N_k=0 Gsuassians will still survive
}
//==========================================================================
void GaussianMixtureModel::killMixture(void)
{
	N_k=0.0;
	alpha_k=0.0;
	for(int ii=0;ii<dimsImage;ii++)
	{
		m_o(ii)=-1e32;
		m_k(ii)=-1e32;
	}
}

//==========================================================================
void GaussianMixtureModel::copyPriors()
{
	m_k=m_o;
	beta_k=beta_o;
	nu_k=nu_o;
	W_k=W_o;
	alpha_k=alpha_k;//this value does not need to be altered
}

//=======================================================================
double GaussianMixtureModel::volumeGaussian()
{
	//compute eigenvalue decomposition
	SelfAdjointEigenSolver<Matrix<double,dimsImage,dimsImage> > sigma1eig(W_k*nu_k);
	switch(dimsImage)
	{
	case 3:
		return ellipsoidVolumeConstant/sqrt(sigma1eig.eigenvalues().prod());//vol=(4/3)Pi*a*b*c
		break;
	case 2:
		return ellipseAreaConstant/sqrt(sigma1eig.eigenvalues().prod());//area=Pi*a*b
		break;
	default:
		cout<<"ERROR: ellipsid volume is not prepared for this number of dimensions"<<endl;
		exit(3);
		return 0.0;
	}

}

//============================================================================================
void GaussianMixtureModel::motionUpdateKalman()
{
	mu_KF=psi*mu_KF;
	P_KF=(psi*P_KF*psi.transpose())+Q;

	//-----------------------print out debugging---------
	/*
	cout<<"DEBUGGING: MCMC::motionUpdateKalman"<<endl;
	cout<<"psi=[\n"<<psi<<"\n];"<<endl;
	cout<<"Q=[\n"<<Q<<"\n];"<<endl;
	*/
	//-----------------------end debug-----------------
}
//=================================================================================================================
void GaussianMixtureModel::informationUpdateKalman()
{
	if(dimsImage>4)
	{
		//gain2.ldlt().solveInPlace(gain);//calculate the inverse efficiently using the fact that covariance matrices are symmetric
		cout<<"ERROR: code is not ready for more than an observation vector of 4 dimensions. Change it to use cholesky decomposition"<<endl;
		exit(2);
	}

	MatrixState2Obs gain=Mobs*P_KF;
	//covM=(beta_k*nu_k*W_k).inverse();
	MatrixObs gain2=((gain*Mobs.transpose())+(W_k.inverse()/(beta_k*nu_k))).inverse();

	//obs=m_k
	mu_KF+=(gain.transpose()*gain2*(m_k-Mobs*mu_KF));
	P_KF-=(gain.transpose()*gain2*gain);
}
//==================================================================================================================
void GaussianMixtureModel::pixels2units(float scaleOrig[dimsImage])
{
	float aux;

	for(int ii=0;ii<dimsImage;ii++)
	{
		scale[ii]=1.0f;
		aux=scaleOrig[ii];
		m_k(ii)*=aux;
		m_o(ii)*=aux;
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_k(ii,jj)/=(aux*scaleOrig[jj]);
			W_o(ii,jj)/=(aux*scaleOrig[jj]);
		}
	}
}

//==================================================================================================================
void GaussianMixtureModel::units2pixels(float scaleOrig[dimsImage])
{
	float aux;

	for(int ii=0;ii<dimsImage;ii++)
	{
		aux=scaleOrig[ii];
		scale[ii]=aux;//juts so I never forget to do it
		m_k(ii)/=aux;
		m_o(ii)/=aux;
		for(int jj=0;jj<dimsImage;jj++)
		{
			W_k(ii,jj)*=(aux*scaleOrig[jj]);
			W_o(ii,jj)*=(aux*scaleOrig[jj]);
		}
	}
}
