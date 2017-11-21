/*
 * variationalInference.cpp
 *
 */

#include "variationalInference.h"
#include "kdtree.cpp"
#include "Utils/MultinormalCDF.h"
#include "Utils/WishartCDF.h"
#if !(defined(_WIN32) || defined(_WIN64))
	#include <unistd.h>
#endif
#include <time.h>
#include <algorithm>
#include <map>
#include "external/Nathan/tictoc.h"

#if defined(_WIN32) || defined(_WIN64)
#include <stdio.h>
#include <process.h>
#define popen _popen
#define getpid _getpid
#define pclose _pclose
#endif


#ifndef ROUND
#define ROUND(x) (floor(x+0.5))
#endif

//@warning Peak memory use is size(img)*(4+4+8+8) bytes
double updateResponsibilities(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r,double thrSignal)
{
	double numSamples=0.0;
	//update all the centers for each Gaussian to calculate tree apropiately
	for(unsigned int kk=0;kk<r.K;kk++) vecGM[kk]->updateCenter();

	//generate kdtree for Gaussian
	KDTree<GaussianMixtureModel> kdtreeGaussian(dimsImage);
	kdtreeGaussian.balanceTree(vecGM);//this function rearranges the order of elements in vecGM

	float cc[dimsImage];
	float radius;
	mylib::Coordinate *coord;
	mylib::Dimn_Type *ccAux;
	GaussianMixtureModel *GM;
	double aux;
	Matrix<double,dimsImage,1> x_n;


	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	//reset all the values to zero: we don't need to reset memory (slow). Just reset counter
	//memset(r.R_nk->x,0,sizeof(double)*r.R_nk->nzmax);
	//memset(r.R_nk->i,0,sizeof(double)*r.R_nk->nzmax);
	//memset(r.R_nk->p,0,sizeof(double)*r.R_nk->nzmax);
	r.R_nk->nz=0;//number of entries in triplet matrix

	double totalAlpha=0.0;
	for(unsigned int ii=0;ii<vecGM.size();ii++) totalAlpha+=vecGM[ii]->alpha_k;


	//precompute some quantities that are independent of location
	double *expectedLogResponsivity=new double[vecGM.size()];
	double *expectedLogDetCovariance=new double[vecGM.size()];
	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		if(vecGM[kk]->isDead()==false)
		{
			expectedLogDetCovariance[vecGM[kk]->id]=vecGM[kk]->expectedLogDetCovariance();
			expectedLogResponsivity[vecGM[kk]->id]=vecGM[kk]->expectedLogResponsivity(totalAlpha);
		}else{
			expectedLogDetCovariance[vecGM[kk]->id]=-1e32;
			expectedLogResponsivity[vecGM[kk]->id]=-1e32;
		}
	}

	for(unsigned long long int n=0;n<r.N;n++)
	{
		if(imgData[n]>thrSignal)//this pixel is considered signal
		{
			numSamples+=imgData[n];

			//find closest maxGaussiansPerVoxel Gaussians
			coord=mylib::Idx2CoordA(img,n);
			ccAux=(mylib::Dimn_Type*)coord->data;
			for(int ii=0;ii<dimsImage;ii++) cc[ii]=(float)ccAux[ii];//transform index 2 coordinates
			for(int ii=0;ii<coord->dims[0];ii++) x_n(ii)=ccAux[ii];
			mylib::Free_Array(coord);

			//best idea so far: download code for Cuda from CVPR 2008
			kdtreeGaussian.findNearest(vecGM,cc,maxGaussiansPerVoxel,1e32,&radius);

			while(!kdtreeGaussian.nearest_queue.empty())
			{
				//calculate r_nk according to equation 10.67
				GM=kdtreeGaussian.nearest_queue.top();


				double aux1=expectedLogResponsivity[GM->id];
				double aux2=expectedLogDetCovariance[GM->id];
				double aux3=GM->expectedMahalanobisDistance(x_n);
				aux=(aux1+0.5*aux2-0.5*aux3);

				//aux=imgData[n]*(expectedLogResponsivity[GM->id]+0.5*expectedLogDetCovariance[GM->id]-0.5*GM->expectedMahalanobisDistance(x_n));

				//to avoid computing too many exponentials
				if(aux>-100)//exp(-100)=3.7201e-44. We consider zero otherwise
				{
					aux=exp(aux);
					cs_entry(r.R_nk,n,GM->id,aux);
				}

				/*
				if(cs_entry(r.R_nk,n,GM->id,aux)==0)
				{
					cout<<"ERROR at updateResponsibilities: sparse matrix is not in triplet form"<<endl;
					exit(2);
				}
				*/

				kdtreeGaussian.nearest_queue.pop();
			}
		}else{//this pixel is not considered signal
				//do nothing
		}
	}

	delete []expectedLogDetCovariance;
	delete []expectedLogResponsivity;

	//----------------------------------
	/*
	cout<<"Looking for errors"<<endl;
	int *hist=new int[r.K];
	memset(hist,0,sizeof(int)*r.K);
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		hist[r.R_nk->p[ii]]++;
	}
	for(unsigned int kk=0;kk<r.K;kk++) cout<<hist[kk]<<";";
	cout<<endl;
	delete[] hist;
	*/
	//---------------------------------
	/*
	//normalize all the rows. r stores values of consecutive n->I don't need to allocate memory: THIS IS NOT TRUE AFTER SPLITTING THE GAUSSIANS!!!
	int pIni=0;//beginning and end pointer for each
	double rowNorm_=0.0;
	int currentN=r.R_nk->i[0];
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		if(currentN!=r.R_nk->i[ii])
		{
			if((ii-pIni)==1) r.R_nk->x[pIni]=1.0;//only one possibility for this element
			else for(int jj=pIni;jj<ii;jj++) r.R_nk->x[jj]/=rowNorm_;
			//reset values
			rowNorm_=0.0;
			currentN=r.R_nk->i[ii];
			pIni=ii;
		}else{
			rowNorm_+=r.R_nk->x[ii];
		}
	}
	*/
	//normalize all the rows TODO r stores values of consecutive n->I don't need to allocate memory
		double *rowNorm=new double[r.N];
		memset(rowNorm,0,sizeof(double)*r.N);
		for(int ii=0;ii<r.R_nk->nz;ii++)
			rowNorm[r.R_nk->i[ii]]+=r.R_nk->x[ii];
		for(int ii=0;ii<r.R_nk->nz;ii++)
		{
			if(rowNorm[r.R_nk->i[ii]]>0)
				r.R_nk->x[ii]/=rowNorm[r.R_nk->i[ii]];
		}
		delete[] rowNorm;

	//reorder vecGM
	sort(vecGM.begin(),vecGM.end(),GaussianMixtureModelPtrComp);
	kdtreeGaussian.needRebalance=true;

	//release memory

	return numSamples;
}



//@warning Peak memory use is size(img)*(4+4+8+8) bytes
void updateResponsibilitiesWithParticles(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r,double thrSignal)
{
	//update all the centers for each Gaussian to calculate tree apropiately
	for(unsigned int kk=0;kk<r.K;kk++) vecGM[kk]->updateCenter();

	//generate kdtree for Gaussian
	KDTree<GaussianMixtureModel> kdtreeGaussian(dimsImage);
	kdtreeGaussian.balanceTree(vecGM);//this function rearranges the order of elements in vecGM

	float cc[dimsImage];
	float radius;
	mylib::Coordinate *coord;
	mylib::Dimn_Type *ccAux;
	GaussianMixtureModel *GM;
	double aux;
	Matrix<double,dimsImage,1> x_n;

	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	//reset all the values to zero
	//memset(r.R_nk->x,0,sizeof(double)*r.R_nk->nzmax);
	//memset(r.R_nk->i,0,sizeof(double)*r.R_nk->nzmax);
	//memset(r.R_nk->p,0,sizeof(double)*r.R_nk->nzmax);
	r.R_nk->nz=0;//number of entries in triplet matrix

	double totalAlpha=0.0;
	for(unsigned int ii=0;ii<vecGM.size();ii++) totalAlpha+=vecGM[ii]->alpha_k;


	//precompute some quantities that are independent of location
	double *expectedLogResponsivity=new double[vecGM.size()];
	double *expectedLogDetCovariance=new double[vecGM.size()];
	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		//expectedLogDetCovariance[vecGM[kk]->id]=vecGM[kk]->expectedLogDetCovariance();
		expectedLogDetCovariance[vecGM[kk]->id]=vecGM[kk]->muLambdaExpectation(&logDeterminant,NULL);

		expectedLogResponsivity[vecGM[kk]->id]=vecGM[kk]->expectedLogResponsivity(totalAlpha);
	}

	for(unsigned long long int n=0;n<r.N;n++)
	{
		if(imgData[n]>thrSignal)//this pixel is considered signal
		{
			//find closest maxGaussiansPerVoxel Gaussians
			coord=mylib::Idx2CoordA(img,n);
			ccAux=(mylib::Dimn_Type*)coord->data;
			for(int ii=0;ii<dimsImage;ii++) cc[ii]=(float)ccAux[ii];//transform index 2 coordinates
			mylib::Free_Array(coord);

			kdtreeGaussian.findNearest(vecGM,cc,maxGaussiansPerVoxel,1e32,&radius);

			while(!kdtreeGaussian.nearest_queue.empty())
			{
				//calculate r_nk according to equation 10.67
				GM=kdtreeGaussian.nearest_queue.top();
				for(int ii=0;ii<coord->dims[0];ii++) x_n(ii)=ccAux[ii];

				double aux1=expectedLogResponsivity[GM->id];
				double aux2=expectedLogDetCovariance[GM->id];
				//double aux3=GM->expectedMahalanobisDistance(x_n);
				double aux3=GM->muLambdaExpectation(&mahalanobisDistance,&x_n);
				aux=(aux1+0.5*aux2-0.5*aux3);

				//aux=imgData[n]*(expectedLogResponsivity[GM->id]+0.5*expectedLogDetCovariance[GM->id]-0.5*GM->expectedMahalanobisDistance(x_n));

				//to avoid computing too many exponentials
				if(aux>-100)//exp(-100)=3.7201e-44. We consider zero otherwise
				{
					aux=exp(aux);
					cs_entry(r.R_nk,n,GM->id,aux);
				}

				/*
				if(cs_entry(r.R_nk,n,GM->id,aux)==0)
				{
					cout<<"ERROR at updateResponsibilities: sparse matrix is not in triplet form"<<endl;
					exit(2);
				}
				*/

				kdtreeGaussian.nearest_queue.pop();
			}
		}else{//this pixel is not considered signal
				//do nothing
		}
	}

	delete []expectedLogDetCovariance;
	delete []expectedLogResponsivity;

	//----------------------------------
	/*
	cout<<"Looking for errors"<<endl;
	int *hist=new int[r.K];
	memset(hist,0,sizeof(int)*r.K);
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		hist[r.R_nk->p[ii]]++;
	}
	for(unsigned int kk=0;kk<r.K;kk++) cout<<hist[kk]<<";";
	cout<<endl;
	delete[] hist;
	*/
	//---------------------------------

	//normalize all the rows
	double *rowNorm=new double[r.N];
	memset(rowNorm,0,sizeof(double)*r.N);
	for(int ii=0;ii<r.R_nk->nz;ii++)
		rowNorm[r.R_nk->i[ii]]+=r.R_nk->x[ii];
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		if(rowNorm[r.R_nk->i[ii]]>0)
			r.R_nk->x[ii]/=rowNorm[r.R_nk->i[ii]];
	}
	delete[] rowNorm;


	//reorder vecGM
	sort(vecGM.begin(),vecGM.end(),GaussianMixtureModelPtrComp);
	kdtreeGaussian.needRebalance=true;

	//release memory

}

//==========================================================================================================
void updateGaussianParameters(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r)
{

	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
		if(img->type!=8)
		{
			cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
			exit(10);
		}

	unsigned int K=vecGM.size();

	for(unsigned int kk=0;kk<K;kk++) vecGM[kk]->N_k=0.0;//reset values
	vector< Matrix<double,dimsImage,1> > x_k(K,Matrix<double,dimsImage,1>::Zero());
	vector< Matrix<double,dimsImage,dimsImage> > S_k(K,Matrix<double,dimsImage,dimsImage>::Zero());


	int n=0,k=0,nOld=-1;
	mylib::Coordinate *coord;
	mylib::Dimn_Type *ccAux;
	Matrix<double,dimsImage,1> x_n;
	double aux;

	TicTocTimer tt=tic();

	//first pass to calculate mean
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		n=r.R_nk->i[ii];
		k=r.R_nk->p[ii];
		if(nOld!=n)//speed up since r.R_nk->i is sorted
		{
			coord=mylib::Idx2CoordA(img,n);
			ccAux=(mylib::Dimn_Type*)coord->data;
			for(int jj=0;jj<coord->dims[0];jj++) x_n(jj)=ccAux[jj];
			Free_Array(coord);
			nOld=n;
		}

		aux=(imgData[n]*r.R_nk->x[ii]);
		vecGM[k]->N_k+=aux;
		x_k[k]+=aux*x_n;


	}
	for(unsigned int kk=0;kk<K;kk++)
	{
		if(vecGM[kk]->N_k>0.0) x_k[kk]/=vecGM[kk]->N_k;
	}

	//second pass to calculate variance
	nOld=-1;
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		n=r.R_nk->i[ii];
		k=r.R_nk->p[ii];
		if(nOld!=n)//speed up since r.R_nk->i is sorted
		{
			coord=mylib::Idx2CoordA(img,n);
			ccAux=(mylib::Dimn_Type*)coord->data;
			for(int jj=0;jj<coord->dims[0];jj++) x_n(jj)=ccAux[jj];
			Free_Array(coord);
			nOld=n;
		}
		aux=(imgData[n]*r.R_nk->x[ii]);
		S_k[k]+=aux*(x_n-x_k[k])*(x_n-x_k[k]).transpose();
	}
	for(unsigned int kk=0;kk<K;kk++)
		if(vecGM[kk]->N_k>0.0)
			S_k[kk]/=vecGM[kk]->N_k;

	//cout<<"Time to calculate S_k,X_k and N_k="<<toc(&tt)<<endl;

	//update mixture of Gaussians parameters
	for(unsigned int kk=0;kk<K;kk++)
	{
		if(vecGM[kk]->isDead()==false)
		{
			vecGM[kk]->updateGaussianParameters(x_k[kk],S_k[kk]);
			//make sure some of the dimensions have not collapsed
			vecGM[kk]->regularizePrecisionMatrix(true);
		}
	}

	//check for cell deaths using pi_k
	double alphaTotal=0.0;
	for(unsigned int kk=0;kk<K;kk++)
		{
			alphaTotal+=vecGM[kk]->alpha_k;
		}

	for(unsigned int kk=0;kk<K;kk++)
	{
		if(vecGM[kk]->alpha_k/alphaTotal<minPi_kForDead && vecGM[kk]->isDead()==false && vecGM[kk]->fixed==false)
		{
			cout<<"WARNING: N_k["<<kk<<"]=0. Cell death"<<endl;
			vecGM[kk]->killMixture();
		}
	}
}

//==========================================================================================================
void updateGaussianParametersWithParticles(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r)
{

	//update all the centers for each Gaussian to calculate tree apropiately
	for(unsigned int kk=0;kk<vecGM.size();kk++) vecGM[kk]->updateCenter();

	//generate kdtree for Gaussian
	KDTree<GaussianMixtureModel> kdtreeGaussian(dimsImage);
	kdtreeGaussian.balanceTree(vecGM);//this function rearranges the order of elements in vecGM

	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
		if(img->type!=8)
		{
			cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
			exit(10);
		}

	unsigned int K=vecGM.size();
	GaussianMixtureModel *GM;
	double auxMRF=0.0,auxProposal=0.0,auxPosterior=0.0;
	mylib::CDF *resample;
	meanPrecisionSample auxResampleVec[numParticles];//TODO: implement resampling without having to duplicate vector of samples
	const double wUniform=1.0/((double)numParticles);
	vector<GaussianMixtureModel*> neighGM;
	neighGM.reserve(maxNeighborsMRF);
	float cc[dimsImage],radius;
	Matrix<double, dimsImage,dimsImage> auxSigma;
	double weightAux[numParticles];
	double sumWeights=0.0;


	//initialize sampling methods
	mylib::uint32 seed;
	mylib::CDF *pos=mylib::Uniform_CDF(0,((double)(img->size-1)));
	mylib::Seed_CDF(pos,getpid()+time(NULL));
	seed=100*imgData[(int)(mylib::Sample_CDF(pos))];
	seed*=65536;//move the upper 4 bits
	seed+=100*imgData[(int)(mylib::Sample_CDF(pos))];
	mylib::Free_CDF(pos);

	auxSigma=Matrix<double,dimsImage,dimsImage>::Identity();
	Wishartdev *wishartCDF=new Wishartdev(dimsImage+10.0,auxSigma,seed++);//bogus initialization
	Multinormaldev *multinormalCDF=new Multinormaldev(vecGM[0]->m_k,auxSigma,seed++);//bogus initialization
	Wishartdev *wishartPosterior=new Wishartdev(dimsImage+10.0,auxSigma,seed++);//bogus initialization
	Multinormaldev *multinormalPosterior=new Multinormaldev(vecGM[0]->m_k,auxSigma,seed++);//bogus initialization
	for(unsigned int kk=0;kk<K;kk++)
	{
		GM=vecGM[kk];
		wishartPosterior->resetParameters(GM->nu_k,GM->W_k);
		auxSigma=(GM->nu_k*GM->W_k).inverse();
		multinormalPosterior->resetParameters(GM->m_k,auxSigma);
		//construct MRF
		for(int ii=0;ii<dimsImage;ii++)cc[ii]=GM->m_k(ii);
		kdtreeGaussian.findNearest(vecGM,cc,maxNeighborsMRF+1,maxDistThrMRF,&radius);//TODO: based neighbors on voronois tesselation

		neighGM.clear();
		while(!kdtreeGaussian.nearest_queue.empty())
		{
			//save neighbors in the MRF graph
			if(kdtreeGaussian.nearest_queue.top()!=GM)//to avoid including ourselves
				neighGM.push_back(kdtreeGaussian.nearest_queue.top());
			kdtreeGaussian.nearest_queue.pop();
		}

		auxSigma=GM->W_k*GM->nu_k/meanPrecisionSample::nuProposal;
		wishartCDF->resetParameters(meanPrecisionSample::nuProposal,auxSigma);//so expected value is GM->w_k
		//sampling importance resampling.
		//We use predetermined parameters for the proposal distribution to make sure it is laxed enough to try different options
		sumWeights=0.0;
		for(int ii=0;ii<numParticles;ii++)
		{
			//propose new solution
			wishartCDF->sample(GM->muLambdaSamples[ii].lambda_k);
			//cout<<GM->muLambdaSamples[ii].lambda_k<<endl;
			//cout<<GM->W_k*GM->nu_k<<endl;
			auxSigma=(GM->muLambdaSamples[ii].lambda_k*meanPrecisionSample::betaProposal).inverse();
			//cout<<auxSigma<<endl;
			multinormalCDF->resetParameters(GM->m_k,auxSigma);
			multinormalCDF->sample(GM->muLambdaSamples[ii].mu_k);
			//calculate weight of the new sample across neighbors in the MRF graph
			auxMRF=0.0;
			for(unsigned int nn=0;nn<neighGM.size();nn++)
			{
				GM->muLambdaSamples[ii].w=GM->sigmaDist_o;
				auxMRF+=neighGM[nn]->muLambdaExpectation(&pairwiseMRF,&(GM->muLambdaSamples[ii]));
			}
			if(isNaN(auxMRF))
				GM->muLambdaSamples[ii].w=0.0;//for proposal that are forbidden (collision between two particles)
			else//I need to include the ratio between posterior and proposal distribution as part of teh importance sampling
			{
				//eval probability from posterior
				auxPosterior=wishartPosterior->evalLog(GM->muLambdaSamples[ii].lambda_k)*multinormalPosterior->eval(GM->muLambdaSamples[ii].mu_k);;
				//eval probability from proposal
				auxProposal=wishartCDF->evalLog(GM->muLambdaSamples[ii].lambda_k)*multinormalCDF->eval(GM->muLambdaSamples[ii].mu_k);
				GM->muLambdaSamples[ii].w=exp(auxMRF+auxPosterior-auxProposal);//since the proposal is the EM without MRF priors, the ratio in importance sampling is just the MRF expectation prior
			}

			weightAux[ii]=GM->muLambdaSamples[ii].w;
			sumWeights+=weightAux[ii];
		}

		//debug--------------------------------
		stringstream itoa;
		itoa<<kk;
		GM->writeParticlesForMatlab("/Users/amatf/TrackingNuclei/tmp/debugParticlesProposal"+itoa.str()+".txt");
		//------------------------------------------------------------------------------
		if(sumWeights<1e-12)
		{
			cout<<"ERROR: Sum weights="<<sumWeights<<endl;
			cout<<"It probably means that the proposal is too tight and all the proposed shapes intersect with other nuclei"<<endl;
			exit(10);
		}

		//resample based on weights
		resample=mylib::Bernouilli_CDF(numParticles,weightAux);
		mylib::Seed_CDF(resample,seed++);
		memcpy(auxResampleVec,GM->muLambdaSamples,sizeof(meanPrecisionSample)*numParticles);
		int pp=0;
		for(int ii=0;ii<numParticles;ii++)
		{
			pp=(int)(mylib::Sample_CDF(resample));
			pp--;
			GM->muLambdaSamples[ii]=auxResampleVec[pp];
			GM->muLambdaSamples[ii].w=wUniform;
		}
		mylib::Free_CDF(resample);

	}
	delete wishartCDF;
	delete multinormalCDF;


	//reorder vecGM
	sort(vecGM.begin(),vecGM.end(),GaussianMixtureModelPtrComp);
	kdtreeGaussian.needRebalance=true;
}


double calculateLogLikelihood(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r)
{
	double logLikelihood=0.0;

	//compute pi_k for each Gaussian
	double totalAlpha=0.0;
	for(unsigned int ii=0;ii<vecGM.size();ii++) totalAlpha+=vecGM[ii]->alpha_k;

	//precompute normal distributions and pi_k
	vector<Multinormaldev*> vecGaussian(vecGM.size());
	vector<double> vecPi_k(vecGM.size());
	Matrix<double,dimsImage,dimsImage> auxSigma;
	for(unsigned int ii=0;ii<vecGM.size();ii++)
	{
		//vecPi_k[ii]=vecGM[ii]->N_k/N;  //This is fo rthe standard EM without priors
		vecPi_k[ii]=vecGM[ii]->alpha_k/totalAlpha;//check equation (10.69) from Bishop
		if( vecGM[ii]->W_k(dimsImage-1,dimsImage-1) < 1e-8 )//it means we have a 2D ellipsoid->just regularize W with any value (we will ignore it in the putput anyway)
		{
			vecGM[ii]->W_k(dimsImage-1,dimsImage-1) = 0.5 * (vecGM[ii]->W_k(0,0) + vecGM[ii]->W_k(1,1));
		}
		auxSigma=(vecGM[ii]->W_k*vecGM[ii]->nu_k).inverse();
		vecGaussian[ii]=new Multinormaldev(vecGM[ii]->m_k,auxSigma,0);
	}

	//iterate over non-zero elements
	int n=0,k=0,nOld=-1;
	Matrix<double,dimsImage,1> x_n;
	mylib::Coordinate *coord;
	mylib::Dimn_Type *ccAux;
	double *likelihoodPerPixel=new double[img->size];
	memset(likelihoodPerPixel,0,sizeof(double)*(img->size));
	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		n=r.R_nk->i[ii];
		k=r.R_nk->p[ii];
		if(n!=nOld)//speed up since r.R_nk->i is sorted
		{
			coord=mylib::Idx2CoordA(img,n);
			ccAux=(mylib::Dimn_Type*)coord->data;
			for(int jj=0;jj<dimsImage;jj++) x_n(jj)=(double)(ccAux[jj]);//transform index 2 coordinates
			mylib::Free_Array(coord);
			nOld=n;
		}
		likelihoodPerPixel[n]+=vecPi_k[k]*vecGaussian[k]->eval(x_n);
	}

	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	for(int ii=0;ii<img->size;ii++)
	{
		if(likelihoodPerPixel[ii]>0) logLikelihood+=imgData[ii]*log(likelihoodPerPixel[ii]);
	}


	//----------------------debug-----------------------
	/*
	ofstream outLikelihood("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1311858269_CPU/debugLikelihoodPerPixel.m");
	outLikelihood<<"likelihoodPerPixel=[";
	for(int ii=0;ii<img->size;ii++) outLikelihood<<likelihoodPerPixel[ii]<<" ";
	outLikelihood<<"];"<<endl;
	outLikelihood<<"imgData=[";
	for(int ii=0;ii<img->size;ii++) outLikelihood<<imgData[ii]<<" ";
	outLikelihood<<"];"<<endl;
	outLikelihood.close();
	exit(0);
	*/
	//---------------------------------------------------


	//release memory
	delete[] likelihoodPerPixel;
	for(unsigned int kk=0;kk<vecGaussian.size();kk++) delete vecGaussian[kk];
	vecGaussian.clear();

	return logLikelihood;
}

double calculateLogLikelihoodWithParticles(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r)
{
	//TODO
	return 0.0;
}


GaussianMixtureModel* splitGaussian(vector<GaussianMixtureModel*> &vecGM, responsibilities &r,int pos)
{
	if(pos>=(int)vecGM.size() || pos<0)
	{
		cout<<"ERROR: at splitGaussian(vector<GaussianMixtureModel*> &vecGM, responsibilities &r,unsigned int plos) trying to split Gaussian with pos larger than number of Gaussians"<<endl;
		exit(2);
	}

	r.K++;
	r.R_nk->n++;
	int nz=r.R_nk->nz;
	//update reponsibilities
	for(int ii=0;ii<nz;ii++)
	{
		if(r.R_nk->p[ii]==pos)
		{
			r.R_nk->x[ii]*=0.5;
			cs_entry(r.R_nk,r.R_nk->i[ii],r.R_nk->n-1,r.R_nk->x[ii]);
		}
	}
	//update Gaussian mixture
	vecGM.push_back(new GaussianMixtureModel());
	vecGM[pos]->splitGaussian(vecGM.back(),vecGM.size()-1);
	return vecGM.back();//returns pointer of the new Gaussian
}

void deleteGaussian(vector<GaussianMixtureModel*> &vecGM, responsibilities &r,int pos)
{
	if(pos>=(int)vecGM.size() || pos<0)
	{
		cout<<"ERROR: at deleteGaussian(vector<GaussianMixtureModel*> &vecGM, responsibilities &r,unsigned int plos) trying to split Gaussian with pos larger than number of Gaussians"<<endl;
		exit(2);
	}

	r.K--;//reduce number of mixtures
	r.R_nk->n--;
	int nz=r.R_nk->nz;
	//update reponsibilities
	for(int ii=0;ii<nz;ii++)
	{
		if(r.R_nk->p[ii]>pos)
		{
			r.R_nk->p[ii]--;
		}
		else if(r.R_nk->p[ii]==pos)
		{
			r.R_nk->x[ii]=0.0;//it is the fastest way to remove this element
			r.R_nk->p[ii]=0;//to make sure it does not point beyond r.K
		}
	}
	//update Gaussian mixture
	for(unsigned int kk=pos+1;kk<vecGM.size();kk++) vecGM[kk]->id--;
	delete vecGM[pos];
	vecGM.erase(vecGM.begin()+pos);
}

//=======================================================================================================================
void calculateLocalKullbackDiversity(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, const responsibilities &r)
{
	//precompute normal distributions
	vector<Multinormaldev*> vecGaussian(vecGM.size());
	Matrix<double,dimsImage,dimsImage> auxSigma;
	for(unsigned int ii=0;ii<vecGM.size();ii++)
	{
		if( vecGM[ii]->W_k(dimsImage-1,dimsImage-1) < 1e-8 )//it means we have a 2D ellipsoid->just regularize W with any value (we will ignore it in the putput anyway)
		{
			vecGM[ii]->W_k(dimsImage-1,dimsImage-1) = 0.5 * (vecGM[ii]->W_k(0,0) + vecGM[ii]->W_k(1,1));
		}
		auxSigma=(vecGM[ii]->W_k*vecGM[ii]->nu_k).inverse();
		vecGaussian[ii]=new Multinormaldev(vecGM[ii]->m_k,auxSigma,0);
		vecGM[ii]->splitScore=0.0;//reset split score values
		if(vecGM[ii]->id!=(int)ii)
		{
			cout<<"ERROR: at calculateLocalKullbackDiversity we need vecGM[kk]->id==kk "<<endl;//crucial for line (*)
			exit(2);
		}
	}

	//iterate over non-zero elements
	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	int n,k;
	double aux;
	Matrix<double,dimsImage,1> x_n;
	mylib::Coordinate *coord;
	mylib::Dimn_Type *ccAux;

	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		n=r.R_nk->i[ii];
		k=r.R_nk->p[ii];
		coord=mylib::Idx2CoordA(img,n);
		ccAux=(mylib::Dimn_Type*)coord->data;
		for(int jj=0;jj<dimsImage;jj++) x_n(jj)=(double)(ccAux[jj]);//transform index 2 coordinates
		mylib::Free_Array(coord);

		aux=r.R_nk->x[ii]*imgData[n];
		//see notebook May 25th 2011 for more details
		if(aux>0.0) vecGM[k]->splitScore+=aux*(log(aux)-vecGaussian[k]->evalLog(x_n)); //line (*)
	}


	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		if(vecGM[kk]->N_k>0.0)
		{
			vecGM[kk]->splitScore/=vecGM[kk]->N_k;//normalize
			vecGM[kk]->splitScore-=log(vecGM[kk]->N_k);//add logarithm,
		}
	}

	//release memory
	for(unsigned int kk=0;kk<vecGaussian.size();kk++) delete vecGaussian[kk];
	vecGaussian.clear();
}

//=======================================================================================================================
double calculatePairwiseLocalKullbackDiversity(mylib::Array *img,GaussianMixtureModel* GM1, GaussianMixtureModel* GM2, const responsibilities &r)
{
	//precompute normal distributions
	Matrix<double,dimsImage,dimsImage> auxSigma;

	auxSigma=(GM1->W_k*GM1->nu_k).inverse();
	Multinormaldev* Gaussian1=new Multinormaldev(GM1->m_k,auxSigma,0);
	auxSigma=(GM2->W_k*GM2->nu_k).inverse();
	Multinormaldev* Gaussian2=new Multinormaldev(GM2->m_k,auxSigma,0);

	double pi1=GM1->alpha_k/(GM1->alpha_k+GM2->alpha_k);
	double pi2=1.0-pi1;
	double splitScore=0.0;

	//iterate over non-zero elements
	//TODO make it work for any type of data
	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	double auxR=0.0;
	int n,k,nOld=-1;
	Matrix<double,dimsImage,1> x_n;
	mylib::Coordinate *coord;
	mylib::Dimn_Type *ccAux;


	for(int ii=0;ii<r.R_nk->nz;ii++)
	{
		n=r.R_nk->i[ii];
		k=r.R_nk->p[ii];

		if(k==GM1->id || k==GM2->id)
		{
			//r.R_nk->i is ordered by n->we can make a very efficient insertion
			if(n==nOld)//we have found both for this n
			{
				auxR+=r.R_nk->x[ii]*imgData[n];
				coord=mylib::Idx2CoordA(img,n);
				ccAux=(mylib::Dimn_Type*)coord->data;
				for(int jj=0;jj<dimsImage;jj++) x_n(jj)=(double)(ccAux[jj]);//transform index 2 coordinates
				mylib::Free_Array(coord);

				if(auxR>0.0) splitScore+=auxR*log(auxR/(pi1*Gaussian1->eval(x_n)+pi2*Gaussian2->eval(x_n)));

				auxR=0.0;

			}else{//we have found only one for nOld

				if(auxR>0.0)//process quantity for nOld
				{
					coord=mylib::Idx2CoordA(img,nOld);
					ccAux=(mylib::Dimn_Type*)coord->data;
					for(int jj=0;jj<dimsImage;jj++) x_n(jj)=(double)(ccAux[jj]);//transform index 2 coordinates
					mylib::Free_Array(coord);

					splitScore+=auxR*log(auxR/(pi1*Gaussian1->eval(x_n)+pi2*Gaussian2->eval(x_n)));
				}
				auxR=r.R_nk->x[ii]*imgData[n];
			}
			nOld=n;
		}

	}
	if(auxR>0.0)//process last step
	{
		coord=mylib::Idx2CoordA(img,nOld);
		ccAux=(mylib::Dimn_Type*)coord->data;
		for(int jj=0;jj<dimsImage;jj++) x_n(jj)=(double)(ccAux[jj]);//transform index 2 coordinates
		mylib::Free_Array(coord);

		splitScore+=auxR*log(auxR/(pi1*Gaussian1->eval(x_n)+pi2*Gaussian2->eval(x_n)));
	}

	//finish normalizing distance
	auxR=GM1->N_k+GM2->N_k;
	splitScore/=auxR;
	splitScore-=log(auxR);

	return splitScore;
}

//===============================================================================

double calculateModelSelectionPenalty(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r,double numSamples)
{
	int Knz=0;//number of mixtures with alpha_k>0
	double penalty=0.0;
	double penalty2=0.0;
	double totalAlpha=0.0;

	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		if(vecGM[kk]->isDead()==false)
		{
			Knz++;
			penalty+=log(vecGM[kk]->alpha_k);
			totalAlpha+=vecGM[kk]->alpha_k;
		}
	}

	penalty*=GaussianMixtureModel::dof;
	penalty2=GaussianMixtureModel::dof*log(numSamples/(12.0*totalAlpha))+log(numSamples/12.0)+GaussianMixtureModel::dof+1;

	return 0.5*(penalty+((double)Knz)*penalty2);
}

/*
//log(K!)
double calculateModelSelectionPenalty(mylib::Array *img, vector<GaussianMixtureModel*> &vecGM, responsibilities &r,double numSamples)
{
	int Knz=0;//number of mixtures with alpha_k>0
	double penalty=0.0;

	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		if(vecGM[kk]->isDead()==false)
		{
			Knz++;
		}
	}
	for(int kk=2;kk<=Knz;kk++) penalty+=log(kk);

	return penalty;
}
*/

//==================================================================
bool checkVecGMidIntengrity(vector<GaussianMixtureModel*> &vecGM)
{
	cout<<"DEBUGGING: checking vecGM id integrity"<<endl;
	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		if(vecGM[kk]->id!=(int)kk)
		{
			cout<<"ERROR: vecGM[kk]->id="<<vecGM[kk]->id<<" for kk="<<kk;
			return false;
		}
	}
	return true;
}

//===========================================================
void copy2GMEM_CUDA(GaussianMixtureModel *GM,GaussianMixtureModelCUDA *GMCUDAtemp)
{

	GMCUDAtemp->beta_k=GM->beta_k;
	GMCUDAtemp->nu_k=GM->nu_k;
	GMCUDAtemp->alpha_k=GM->alpha_k;
	memcpy(GMCUDAtemp->m_k,GM->m_k.data(),dimsImage*sizeof(double));
	int idx=0;
	for(int ii=0;ii<dimsImage;ii++)
		for(int jj=ii;jj<dimsImage;jj++)
		{
			GMCUDAtemp->W_k[idx]=GM->W_k(ii,jj);
			GMCUDAtemp->W_o[idx++]=GM->W_o(ii,jj);
		}

	//priors
	GMCUDAtemp->beta_o=GM->beta_o;
	GMCUDAtemp->nu_o=GM->nu_o;
	GMCUDAtemp->alpha_o=GM->alpha_o;
	memcpy(GMCUDAtemp->m_o,GM->m_o.data(),dimsImage*sizeof(double));

	//indicators
	GMCUDAtemp->fixed=GM->fixed;

	GMCUDAtemp->splitScore=GM->splitScore;

	//copy supervoxel assignment
	GMCUDAtemp->supervoxelNum = GM->supervoxelIdx.size();
	for(unsigned int ii = 0;ii< GM->supervoxelIdx.size();ii++)
		GMCUDAtemp->supervoxelIdx[ii] = GM->supervoxelIdx[ii];
}
//===========================================================
void copyFromGMEM_CUDA(GaussianMixtureModelCUDA *GMCUDAtemp,GaussianMixtureModel *GM) 
{

	GM->beta_k=GMCUDAtemp->beta_k;
	GM->nu_k=GMCUDAtemp->nu_k;
	GM->alpha_k=GMCUDAtemp->alpha_k;
	memcpy(GM->m_k.data(),GMCUDAtemp->m_k,dimsImage*sizeof(double));
	int idx=0;
	for(int ii=0;ii<dimsImage;ii++)
	{
		GM->W_k(ii,ii)=GMCUDAtemp->W_k[idx];
		GM->W_o(ii,ii)=GMCUDAtemp->W_o[idx++];
		for(int jj=ii+1;jj<dimsImage;jj++)
		{
			GM->W_k(ii,jj)=GMCUDAtemp->W_k[idx];
			GM->W_o(ii,jj)=GMCUDAtemp->W_o[idx];
			GM->W_k(jj,ii)=GMCUDAtemp->W_k[idx];
			GM->W_o(jj,ii)=GMCUDAtemp->W_o[idx++];
		}
	}
	//priors
	GM->beta_o=GMCUDAtemp->beta_o;
	GM->nu_o=GMCUDAtemp->nu_o;
	GM->alpha_o=GMCUDAtemp->alpha_o;
	memcpy(GM->m_o.data(),GMCUDAtemp->m_o,dimsImage*sizeof(double));

	//indicators
	GM->fixed=GMCUDAtemp->fixed;

	GM->splitScore=GMCUDAtemp->splitScore;

	//update N_k
	GM->updateNk();

	//copy supervoxel assignment
	GM->supervoxelIdx.resize(GMCUDAtemp->supervoxelNum);
	for(unsigned int ii = 0;ii< GM->supervoxelIdx.size();ii++)
		GM->supervoxelIdx[ii] = GMCUDAtemp->supervoxelIdx[ii];
}


//==================================================
//TODO: I should use r_nk (responsibities) to calculate a weighted mean of the flow for each cell
int applyFlowPredictionToNuclei(mylib::Array* imgFlow,mylib::Array* imgFlowMask,vector<GaussianMixtureModel*> &vecGM,bool isForwardFlow)
{
	int winSize[dimsImage]={2,2,1};

	if(imgFlow->type!=mylib::FLOAT32_TYPE)
	{
		cout<<"ERROR: at applyFlowPredictionToNuclei: flowArray expected to be FLOAT32"<<endl;
		return 1;
	}
	if(imgFlow->ndims!=4)
	{
		cout<<"ERROR: at applyFlowPredictionToNuclei: code is not ready for other than 3D"<<endl;
		return 2;
	}

	if(imgFlowMask!=NULL)
	{
		if(imgFlowMask->type!=mylib::UINT8_TYPE)
		{
			cout<<"ERROR: at applyFlowPredictionToNuclei: imgFlowMask expected to be UINT8"<<endl;
			return 3;
		}
		if(imgFlowMask->ndims+1!=imgFlow->ndims)
		{
			cout<<"ERROR: at applyFlowPredictionToNuclei: imgFlowMask and imgFlow have different n-dimensions"<<endl;
			return 5;
		}

		for(int aa=0;aa<imgFlowMask->ndims;aa++)
		{
			if(imgFlow->dims[aa]!=imgFlowMask->dims[aa])
			{
				cout<<"ERROR: at applyFlowPredictionToNuclei: imgFlowMask and imgFlow have different sizes"<<endl;
				return 4;
			}
		}
	}

	mylib::float32* imgFlowPtr=(mylib::float32*)(imgFlow->data);
	mylib::uint8*	imgFlowMaskPtr=NULL;
	if(imgFlowMask!=NULL)
		imgFlowMaskPtr=(mylib::uint8*)(imgFlowMask->data);

	//----------------------------------------debug-------------------------------
	/*
	cout<<"DEBUGGING: writing out flowArray with size ";
	for(int ii=0;ii<imgFlow->ndims;ii++) cout<<imgFlow->dims[ii]<<"x";
	cout<<endl;
	ofstream outBin;
	string filenameOut("debugFlow.bin");
	outBin.open(filenameOut.c_str(),ios::binary | ios::out);
	
	if(outBin.is_open()==false)
	{
		cout<<"ERROR: at mainTestOpticalFlow: flow array file "<<filenameOut<<" could not be written"<<endl;
		return 3;
	}
	outBin.write((char*)(imgFlow->data),sizeof(mylib::float32)*(imgFlow->size));
	outBin.close();
	*/
	//-----------------------------------end debug-----------------------------------

	long long int p[dimsImage],count;
	double flow;
	long long int pos;
	for(unsigned int kk=0;kk<vecGM.size();kk++)
	{
		if(vecGM[kk]->isDead()==true) continue;

		for(int ii=0;ii<dimsImage;ii++)
		{
			p[ii]=(long long int)ROUND(vecGM[kk]->m_k(ii));
		}

		if(imgFlowMask!=NULL)//check if we should apply flow here
		{
			pos=p[0]+imgFlowMask->dims[0]*(p[1]+imgFlowMask->dims[1]*p[2]);
			if(pos>=0 && pos<imgFlowMask->size)
			{
				if(imgFlowMaskPtr[pos]==0) continue;
			}else continue; //it is out of bounds
		}
		for(int ii=0;ii<dimsImage;ii++)
		{
			flow=0.0;
			count=0;
			//calculate avergae flow around nuclei
			//TODO: do this with mylib::frames
			for(long long int zz=p[2]-winSize[2];zz<=p[2]+winSize[2];zz++)
			{
				for(long long int yy=p[1]-winSize[1];yy<=p[1]+winSize[1];yy++)
				{
					for(long long int xx=p[0]-winSize[0];xx<=p[0]+winSize[0];xx++)
					{
						pos=xx+imgFlow->dims[0]*(yy+imgFlow->dims[1]*(zz+imgFlow->dims[2]*ii));
						if(pos>=0 && pos<imgFlow->size)
						{
							flow+=imgFlowPtr[pos];
							count++;
						}
					}
				}
			}
			if(count>0) flow/=(double)(count);
			if(isForwardFlow==true)
			{
				//update flow
				vecGM[kk]->m_k(ii)-=flow;//negative because we calculate flow from t->t+1 (forward in time)
				vecGM[kk]->m_o(ii)-=flow;
			}else{
				//update flow
				vecGM[kk]->m_k(ii)+=flow;//positive because we calculate flow from t+1->t (backwards in time)
				vecGM[kk]->m_o(ii)+=flow;
			}
		}
	}
	return 0;
}
