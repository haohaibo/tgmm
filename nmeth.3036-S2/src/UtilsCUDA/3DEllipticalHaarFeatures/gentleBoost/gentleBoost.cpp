#include "gentleBoost.h"
#include <algorithm>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "incrementalQuantiles.h"


using namespace std;

//ascending sorting
bool compWeightAndSample (weightAndSample a, weightAndSample b)
{
	return a.xSample<b.xSample;
}

//ascending order
int compare (const void * a, const void * b)
{
	if(*(feature*)a < *(feature*)b) return -1;
	else return 1;//we assume equal sign is less than. 
}

inline feature signAmatf(feature x)
{
	if(x<0) return -1.0;
	else return 1.0;
}

ostream& operator<<(ostream& output, const treeStump & p)
{
	output<<"featureNdx="<<p.featureNdx<<endl;
	output<<"a="<<p.a<<endl;
	output<<"b="<<p.b<<endl;
	output<<"th="<<p.th;
	return output;
}
bool operator==(const treeStump & p1, const treeStump & p2)
{
	if( p1.featureNdx != p2.featureNdx)
		return false;
	if( p1.gtChild != p2.gtChild )
		return false;
	if( p1.leChild != p2.leChild )
		return false;
	if( p1.parentIdx != p2.parentIdx )
		return false;
	
	if( fabs(p1.th - p2.th) > 1e-3 )
		return false;
	if( fabs(p1.a - p2.a) > 1e-3 )
		return false;
	if( fabs(p1.b - p2.b) > 1e-3 )
		return false;
	
	return true;
}
bool operator!=(const treeStump & p1, const treeStump & p2)
{
	if( p1.featureNdx != p2.featureNdx)
		return true;
	if( p1.gtChild != p2.gtChild )
		return true;
	if( p1.leChild != p2.leChild )
		return true;
	if( p1.parentIdx != p2.parentIdx )
		return true;
	
	if( fabs(p1.th - p2.th) / fabs(p1.th) > 1e-2 )
		return true;
	if( fabs(p1.a - p2.a) / fabs(p1.a) > 1e-2 )
		return true;
	if( fabs(p1.b - p2.b) / fabs(p1.b) > 1e-2 )
		return true;
	
	return false;
}

//==================================================================================================================================
void gentleTreeBoost(feature *xTrain,feature *yTrain,feature *w, long long int numWeakClassifiers, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int J)
{
	
	treeStump bestWc;
	feature totalW;
	vector<weightAndSample> auxRegression(numSamples);
	
	
	bool deleteW=false;
	if(w==NULL)
	{
		w=new feature[numSamples];
		totalW=1.0/((feature)numSamples);
		for(unsigned long long int ii=0;ii<numSamples;ii++) w[ii]=totalW;		
		deleteW=true;
	}else{//normalize weights
		totalW=0.0;
		for(unsigned long long int ii=0;ii<numSamples;ii++) totalW+=w[ii];
		for(unsigned long long int ii=0;ii<numSamples;ii++) w[ii]/=totalW;
	}
	
	//check that yTrain\in{+1,-1}
	for(unsigned long long int ii=0;ii<numSamples;ii++)
	{
		if(fabs(fabs(yTrain[ii])-1.0)>1e-4)
		{
			cout<<"ERROR: training set labels should be +1 or -1. Code is not ready for other alternatives"<<endl; 
			exit(2);
		}
	}
	
	//sort samples: we do it only once. It takes more memory but it saves a lot of time. Basically sorting algorithm within the loop takes 99% time
	unsigned long long int *sortIdx=new unsigned long long int[numSamples*numFeatures];
	unsigned long long int pos=0,pos2=0;
	for(unsigned long long int ii=0;ii<numFeatures;ii++)
	{
		for(unsigned long long int jj=0;jj<numSamples;jj++)
		{
			auxRegression[jj].xSample=xTrain[pos];
			auxRegression[jj].idx=jj;
			pos++;
		}
		sort(auxRegression.begin(),auxRegression.end(),compWeightAndSample);
		for(unsigned long long int jj=0;jj<numSamples;jj++)
		{
			sortIdx[pos2]=auxRegression[jj].idx;
			pos2++;
		}
	}
	
	classifier.resize(numWeakClassifiers);
	unsigned long long int numSplits=(unsigned long long int)pow(2.0f,(int)J)-1;//number of stumps we need to define the tree
	vector< vector<unsigned long long int> > listIdTree(numSplits);//stores the elemnts at each stage in the tree
	
	for(long long int kk=0;kk<numWeakClassifiers;kk++)
	{
		classifier[kk].clear();
		classifier[kk].resize(numSplits);
		long long int aa=0;
		
		//initialize listIdTree
		listIdTree[0].resize(numSamples);
		for(unsigned long long int jj=0;jj<numSamples;jj++) listIdTree[0][jj]=jj;		
		
		for(unsigned long long int ii=0;ii<J;ii++)
		{
			long long int numLeaves=(long long int)pow(2.0f,(int)ii);
			for(long long int jj=0;jj<numLeaves;jj++)
			{
				fitTreeStumpParallelThreads(xTrain,yTrain,w,numSamples,numFeatures,sortIdx,listIdTree[aa],bestWc);
				long long int gtChildIdx=aa+numLeaves+jj;
				long long int leChildIdx=gtChildIdx+1;//to save time				
				//save stump
				long long int pp=classifier[kk][aa].parentIdx;
				bestWc.parentIdx=pp;
				if(bestWc.featureNdx<0)//sometimes leave is empty
				{
					//undo parent child
					if(pp>=0)
					{
						if(classifier[kk][pp].gtChild==aa)
						{
							classifier[kk][pp].gtChild=-1;
						}else{
							classifier[kk][pp].leChild=-1;
						}
					}
					bestWc.parentIdx=-1;
				}
				else if(ii==J-1)
				{
					bestWc.gtChild=-1;
					bestWc.leChild=-1;
				}else{
					bestWc.gtChild=gtChildIdx;
					bestWc.leChild=leChildIdx;
					//generate new listIdTree elements
					feature *xTrainPtr=&(xTrain[bestWc.featureNdx*numSamples]);
					for(vector<unsigned long long int>::const_iterator iter=listIdTree[aa].begin();iter!=listIdTree[aa].end();++iter)
					{
						if( xTrainPtr[*iter]>bestWc.th ) 
							listIdTree[gtChildIdx].push_back(*iter);
						else 
							listIdTree[leChildIdx].push_back(*iter);
					}
					classifier[kk][gtChildIdx].parentIdx=aa;
					classifier[kk][leChildIdx].parentIdx=aa;
				}
				//release memory
				listIdTree[aa].clear();
				classifier[kk][aa++]=bestWc;
			}
		}
		//evaluate weak classifier and reweight training samples
		feature fm;
		totalW=0;
		treeStump *auxStump;
		for(unsigned long long int ii=0;ii<numSamples;ii++)
		{
			long long int childNode=0;
			while(childNode!=-1)
			{
				auxStump=&(classifier[kk][childNode]);
				if(xTrain[numSamples*(auxStump->featureNdx)+ii]>auxStump->th)
				{
					childNode=auxStump->gtChild;
					if(childNode==-1)//update prediction
					{
						fm=auxStump->a+auxStump->b;
					}
				}
				else{
					childNode=auxStump->leChild; 								
					if(childNode==-1)//update prediction
					{
						fm=auxStump->b;
					}
				}
			}
			
			w[ii]*=exp(-yTrain[ii]*fm);
			totalW+=w[ii];
		}
		//normalize weights
		for(unsigned long long int ii=0;ii<numSamples;ii++) w[ii]/=totalW;
	}
	//release memory
	auxRegression.clear();
	delete[] sortIdx;
	if(deleteW) delete[] w;
};

//==================================================================================================================================
void gradientTreeBoost(feature *xTrain,feature *yTrain, long long int numWeakClassifiers, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int J,lossFunction L)
{
	
	treeStump bestWc;
	vector<weightAndSample> auxRegression(numSamples);
	feature *w=new feature[numSamples];//needed just to be able to call fitStump
	feature *residTrain=new feature[numSamples];//residuals to fit regressions tree
	feature *fmTrain=new feature[numSamples];//cumulative value of boosting (additive) model
	memset(fmTrain,0,sizeof(feature)*numSamples);
	feature auxW=1.0/((feature)numSamples);
	feature delta=1.0;//needed for Huber function: at the beginning residTrain=yTrain->abs(residTrain=1.0 for all
	//check that yTrain\in{+1,-1}
	for(unsigned long long int ii=0;ii<numSamples;ii++)
	{
		w[ii]=auxW;
		if(fabs(fabs(yTrain[ii])-1.0)>1e-4)
		{
			cout<<"ERROR: training set labels should be +1 or -1. Code is not ready for other alternatives"<<endl; 
			exit(2);
		}
	}
	
	//sort samples: we do it only once. It takes more memory but it saves a lot of time. Basically sorting algorithm within the loop takes 99% time
	unsigned long long int *sortIdx=new unsigned long long int[numSamples*numFeatures];
	unsigned long long int pos=0,pos2=0;
	for(unsigned long long int ii=0;ii<numFeatures;ii++)
	{
		for(unsigned long long int jj=0;jj<numSamples;jj++)
		{
			auxRegression[jj].xSample=xTrain[pos];
			auxRegression[jj].idx=jj;
			pos++;
		}
		sort(auxRegression.begin(),auxRegression.end(),compWeightAndSample);
		for(unsigned long long int jj=0;jj<numSamples;jj++)
		{
			sortIdx[pos2]=auxRegression[jj].idx;
			pos2++;
		}
	}
	
	classifier.resize(numWeakClassifiers);
	unsigned long long int numSplits=(unsigned long long int)pow(2.0f,(int)J)-1;//number of stumps we need to define the tree
	vector< vector<unsigned long long int> > listIdTree(numSplits);//stores the elemnts at each stage in the tree
	memcpy(residTrain, yTrain,numSamples*sizeof(feature));//initialize residTrain
	for(long long int kk=0;kk<numWeakClassifiers;kk++)
	{
		classifier[kk].clear();
		classifier[kk].resize(numSplits);
		long long int aa=0;
		
		//initialize listIdTree
		listIdTree[0].resize(numSamples);
		for(unsigned long long int jj=0;jj<numSamples;jj++) listIdTree[0][jj]=jj;		
		
		for(unsigned long long int ii=0;ii<J;ii++)
		{
			long long int numLeaves=(long long int)pow(2.0f,(int)ii);
			for(long long int jj=0;jj<numLeaves;jj++)
			{
				fitTreeStumpParallelThreads(xTrain,residTrain,w,numSamples,numFeatures,sortIdx,listIdTree[aa],bestWc);
				long long int gtChildIdx=aa+numLeaves+jj;
				long long int leChildIdx=gtChildIdx+1;//to save time				
				//save stump
				long long int pp=classifier[kk][aa].parentIdx;
				bestWc.parentIdx=pp;
				if(bestWc.featureNdx<0)//sometimes leave is empty
				{
					//undo parent child
					if(pp>=0)
					{
						if(classifier[kk][pp].gtChild==aa)
						{
							classifier[kk][pp].gtChild=-1;
						}else{
							classifier[kk][pp].leChild=-1;
						}
					}
					bestWc.parentIdx=-1;
				}
				else if(ii==J-1)
				{
					bestWc.gtChild=-1;
					bestWc.leChild=-1;
				}else{
					bestWc.gtChild=gtChildIdx;
					bestWc.leChild=leChildIdx;
					//generate new listIdTree elements
					feature *xTrainPtr=&(xTrain[bestWc.featureNdx*numSamples]);
					for(vector<unsigned long long int>::const_iterator iter=listIdTree[aa].begin();iter!=listIdTree[aa].end();++iter)
					{
						if(xTrainPtr[*iter]>bestWc.th) listIdTree[gtChildIdx].push_back(*iter);
						else listIdTree[leChildIdx].push_back(*iter);
					}
					classifier[kk][gtChildIdx].parentIdx=aa;
					classifier[kk][leChildIdx].parentIdx=aa;
				}
				//find the optimal values a,b according to the loss function (fitting the tree to the residual only gives us the regions)
				leChildIdx=bestWc.leChild;//just to be sure
				if(leChildIdx>=0 && listIdTree[leChildIdx].empty()==false)
				{
					switch (L)
					{
						case L_2://mean
						{
							feature auxMean=0;
							for(vector<unsigned long long int>::const_iterator iter=listIdTree[leChildIdx].begin();iter!=listIdTree[leChildIdx].end();++iter) 
								auxMean+=(yTrain[*iter]-fmTrain[*iter]);
							bestWc.b=auxMean/((feature)listIdTree[leChildIdx].size());
							break;
						}
						case L_1://median TODO: using optimization is probably faster
						{
							feature *auxMedian=new feature[listIdTree[leChildIdx].size()];
							long long int count=0;
							for(vector<unsigned long long int>::const_iterator iter=listIdTree[leChildIdx].begin();iter!=listIdTree[leChildIdx].end();++iter,count++) 
								auxMedian[count]=(yTrain[*iter]-fmTrain[*iter]);
							//qsort(auxMedian,listIdTree[leChildIdx].size(),sizeof(feature),compare);
							//bestWc.b=auxMedian[listIdTree[leChildIdx].size()/2];
							bestWc.b=findL1center(auxMedian,listIdTree[leChildIdx].size());
							delete[] auxMedian;
							break;
						}
						case Huber://requires some optimization TODO
						{
							feature *auxMedian=new feature[listIdTree[leChildIdx].size()];
							long long int count=0;
							for(vector<unsigned long long int>::const_iterator iter=listIdTree[leChildIdx].begin();iter!=listIdTree[leChildIdx].end();++iter,count++) 
								auxMedian[count]=(yTrain[*iter]-fmTrain[*iter]);
							//qsort(auxMedian,listIdTree[leChildIdx].size(),sizeof(feature),compare);
							//bestWc.b=auxMedian[listIdTree[leChildIdx].size()/2];
							bestWc.b=findHubercenter(auxMedian,listIdTree[leChildIdx].size(),delta);
							delete[] auxMedian;
							break;
						}
						default:
						{
							cout<<"ERROR: select loss function is not available"<<endl;
							exit(3);
							break;
						}
					}
				}
				//find the optimal values a,b according to the loss function (fitting the tree to the residual only gives us the regions)
				gtChildIdx=bestWc.gtChild;//just to be sure
				if(gtChildIdx>=0 && listIdTree[gtChildIdx].empty()==false)
				{
					switch (L)
					{
						case L_2://mean
						{
							feature auxMean=0;
							for(vector<unsigned long long int>::const_iterator iter=listIdTree[gtChildIdx].begin();iter!=listIdTree[gtChildIdx].end();++iter) 
								auxMean+=(yTrain[*iter]-fmTrain[*iter]);
							bestWc.a=auxMean/((feature)listIdTree[gtChildIdx].size())-bestWc.b;
							break;
						}
						case L_1://median TODO: using optimization is probably faster
						{
							feature *auxMedian=new feature[listIdTree[gtChildIdx].size()];
							long long int count=0;
							for(vector<unsigned long long int>::const_iterator iter=listIdTree[gtChildIdx].begin();iter!=listIdTree[gtChildIdx].end();++iter,count++) 
								auxMedian[count]=(yTrain[*iter]-fmTrain[*iter]);
							//parabolic approximation tends to be faster and can be reused for any other function
							//qsort(auxMedian,listIdTree[gtChildIdx].size(),sizeof(feature),compare);
							//bestWc.a=auxMedian[listIdTree[gtChildIdx].size()/2]-bestWc.b;
							bestWc.a=findL1center(auxMedian,listIdTree[gtChildIdx].size())-bestWc.b;
							delete[] auxMedian;
							break;
						}
						case Huber://requires some optimization 
						{
							feature *auxMedian=new feature[listIdTree[gtChildIdx].size()];
							long long int count=0;
							for(vector<unsigned long long int>::const_iterator iter=listIdTree[gtChildIdx].begin();iter!=listIdTree[gtChildIdx].end();++iter,count++) 
								auxMedian[count]=(yTrain[*iter]-fmTrain[*iter]);
							//parabolic approximation tends to be faster and can be reused for any other function
							//qsort(auxMedian,listIdTree[gtChildIdx].size(),sizeof(feature),compare);
							//bestWc.a=auxMedian[listIdTree[gtChildIdx].size()/2]-bestWc.b;
							bestWc.a=findHubercenter(auxMedian,listIdTree[gtChildIdx].size(),delta)-bestWc.b;
							delete[] auxMedian;
							break;
						}
						default:
						{
							cout<<"ERROR: select loss function is not available"<<endl;
							exit(3);
							break;
						}
					}
				}
				
				//release memory
				listIdTree[aa].clear();
				classifier[kk][aa++]=bestWc;
			}
		}
		//evaluate weak classifier to estimate residual
		feature fm;
		treeStump *auxStump;
		for(unsigned long long int ii=0;ii<numSamples;ii++)
		{
			long long int childNode=0;
			while(childNode!=-1)
			{
				auxStump=&(classifier[kk][childNode]);
				if(xTrain[numSamples*(auxStump->featureNdx)+ii]>auxStump->th)
				{
					childNode=auxStump->gtChild;
					if(childNode==-1)//update prediction
					{
						fm=auxStump->a+auxStump->b;
					}
				}
				else{
					childNode=auxStump->leChild; 								
					if(childNode==-1)//update prediction
					{
						fm=auxStump->b;
					}
				}
			}
			fmTrain[ii]+=fm;
			
			switch (L)
			{
				case L_2:
					residTrain[ii]=yTrain[ii]-fmTrain[ii];
					break;
				case L_1:
					residTrain[ii]=signAmatf(yTrain[ii]-fmTrain[ii]);
					break;
				case Huber:
					residTrain[ii]=fabs(yTrain[ii]-fmTrain[ii]);
					break;
				default:
					cout<<"ERROR: select loss function is not available"<<endl;
					exit(3);
					break;
			}
		}
		
		//extra computation for Huber function
		if(L==Huber)
		{
			//find quantile
			qsort(residTrain,numSamples,sizeof(feature),compare);
			delta=residTrain[(long long int)(numSamples*Huber_delta)];
			for(unsigned long long int ii=0;ii<numSamples;ii++)
			{
				fm=yTrain[ii]-fmTrain[ii];
				if(fabs(fm)<=delta) residTrain[ii]=fm;
				else residTrain[ii]=delta*signAmatf(fm);
			}
		}
		
	}
	//release memory
	auxRegression.clear();
	delete[] sortIdx;
	delete[] w;
	delete[] residTrain;
	delete[] fmTrain;
};



//----------------------------------------------------------------------------------------------
void boostingTreeClassifier(feature *xTest,feature *Fx, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures)
{

	memset(Fx,0,sizeof(feature)*numSamples);//reset value
	
	for(unsigned long long int cc=0;cc<classifier.size();cc++)
	{		
		vector<treeStump> *weakClassifier=&(classifier[cc]);
		treeStump *auxStump;
		for(unsigned long long int ii=0;ii<numSamples;ii++)
		{
			long long int childNode=0;
			while(childNode!=-1)
			{
				auxStump=&((*weakClassifier)[childNode]);
				if(xTest[numSamples*(auxStump->featureNdx)+ii]>auxStump->th)
				{
					childNode=auxStump->gtChild;
					if(childNode==-1)//update prediction
					{
						Fx[ii]+=auxStump->a+auxStump->b;
					}
				}
				else{
					childNode=auxStump->leChild; 								
					if(childNode==-1)//update prediction
					{
						Fx[ii]+=auxStump->b;
					}
				}
			}
		}
	}
	
}

//----------------------------------------------------------------------------------------------
void boostingTreeClassifierTranspose(feature *xTest,feature *Fx, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures)
{

	memset(Fx,0,sizeof(feature)*numSamples);//reset value
	
	for(unsigned long long int cc=0; cc < classifier.size(); cc++)
	{		
		vector<treeStump> *weakClassifier=&(classifier[cc]);
		treeStump *auxStump;
		long long int offset = 0;
		feature *xTestAux;
		for(unsigned long long int ii=0;ii<numSamples;ii++)
		{
			long long int childNode=0;
			xTestAux = &(xTest[offset]);
			while(childNode!=-1)
			{
				auxStump=&((*weakClassifier)[childNode]);
				//if(xTest[numSamples*(auxStump->featureNdx)+ii]>auxStump->th) //not-transposed
				if(xTestAux[auxStump->featureNdx ] > auxStump->th)
				{
					childNode = auxStump->gtChild;
					if(childNode==-1)//update prediction
					{
						Fx[ii] += (auxStump->a+auxStump->b);
					}
				}
				else{
					childNode=auxStump->leChild; 								
					if(childNode==-1)//update prediction
					{
						Fx[ii] += (auxStump->b);
					}
				}
			}
			
			offset += ( (long long int) numFeatures);
		}
	}
	
}

//--------------------------------------------------------------------------------------------------
//given a collection of samples find the best split 
//In order to be able to reuse the code recursively, we use listId to indicate the points belonging to this split
//numSamples is the total number of samples
void fitTreeStump(feature *xTrain,feature *yTrain,feature *w,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int *sortIdx,const vector<unsigned long long int> &listId,treeStump &bestWc)
{
	//specific case when we do not have any elements (to avoid breaking the code)
	if(listId.empty())
	{
		bestWc.a=0;
		bestWc.b=0;
		bestWc.th=0;
		bestWc.featureNdx=-1;
		bestWc.gtChild=-1;
		bestWc.leChild=-1;
		return;
	}
	//check for special case when all the values of yTrain are the same: the solution to the minimization is undefinded
	feature aux=yTrain[listId[0]];
	bool allEqual=true;
	for(vector<unsigned long long int>::const_iterator iter=listId.begin();iter!=listId.end();++iter)
	{
		if(fabs(aux-yTrain[*iter])>1e-3)
		{
			allEqual=false;
			break;
		}
	}
	if(allEqual)
	{
		//threshold is the maximum value of xTrain[feature=0]
		bestWc.th=-1e32;
		for(vector<unsigned long long int>::const_iterator iter=listId.begin();iter!=listId.end();++iter)
		{
			if(xTrain[*iter]>bestWc.th) 
				bestWc.th=xTrain[*iter];
		}
		bestWc.a=-aux-aux;
		bestWc.b=aux;
		bestWc.th+=1.0;//to give some margin
		bestWc.featureNdx=0;
		return;
	}
	
	feature aPb,b;
	unsigned long long int numSamplesStump=listId.size();
	feature errorAux,errorMin=1e32;
	vector<feature> cumSum(numSamplesStump);
	vector<feature> cumSumW(numSamplesStump);
	unsigned long long int *sortIdxAux=new unsigned long long int[numSamplesStump]; 
	unsigned char *intersection=new unsigned char[numSamples];
	memset(intersection,0,numSamples);
	for(vector<unsigned long long int>::const_iterator iter=listId.begin();iter!=listId.end();++iter) intersection[*iter]='1';//this is the same for all the features
	
	//select best regression stump
	for(unsigned long long int ii=0;ii<numFeatures;ii++)
	{
		/*
		 % [th, a , b] = fitRegressionStump(x, z);
		 % The regression has the form:
		 % z_hat = a * (x>th) + b;
		 %
		 % where (a,b,th) are so that it minimizes the weighted error:
		 % error = sum(w * |z - (a*(x>th) + b)|^2)
		 */
		unsigned long long int *sortIdxPtr=&(sortIdx[ii*numSamples]);
		feature *xTrainPtr=&(xTrain[ii*numSamples]);
		
		//find the subset of sortIdx that is meaningful for listId 
		if(numSamplesStump!=numSamples)//in the first pass we can avoid this part
		{
			unsigned long long int count=0;
			for(unsigned long long int jj=0;jj<numSamples && count<=numSamplesStump;jj++)
			{				
				if(intersection[sortIdxPtr[jj]]=='1') 
					sortIdxAux[count++]=sortIdxPtr[jj];
			}
			sortIdxPtr=sortIdxAux;
			if(count!=numSamplesStump)
			{
				cout<<"ERROR: mismatch between numSamplesStump "<<numSamplesStump<<" and counted samples "<<count<<endl;
				exit(2);
			}
		}
		/*
		//find the subset of sortIdx that is meaningful for listId 
		if(numSamplesStump!=numSamples)//in the first pass we can avoid this part
		{
			unsigned long long int count=0;
			for(unsigned long long int jj=0;jj<numSamples && count<=numSamplesStump;jj++)
			{
				//we assume listId is sorted!
				if(binary_search(listId.begin(),listId.end(),sortIdxPtr[jj]))
					sortIdxAux[count++]=sortIdxPtr[jj];			
			}
			sortIdxPtr=sortIdxAux;
			if(count!=numSamplesStump)
			{
				cout<<"ERROR: mismatch between numSamplesStump and counted samples"<<endl;
				exit(2);
			}
		}
		 */
		//calculate optimal threshold
		unsigned long long int idx=sortIdxPtr[0];
		cumSum[0]=yTrain[idx]*w[idx];
		cumSumW[0]=w[idx];
		for(unsigned long long int jj=1;jj<numSamplesStump;jj++)
		{
			idx=sortIdxPtr[jj];
			cumSum[jj]=cumSum[jj-1]+yTrain[idx]*w[idx];
			cumSumW[jj]=cumSumW[jj-1]+w[idx];
		}
		
		feature partialW=0;
		feature totalCumSum=cumSum[numSamplesStump-1];
		feature totalW=cumSumW[numSamplesStump-1];
		
		for(unsigned long long int jj=0;jj<numSamplesStump-1;jj++)
		{
			idx=sortIdxPtr[jj];
			partialW=cumSumW[jj];
			b=cumSum[jj]/partialW;
			aPb=(totalCumSum-cumSum[jj])/(totalW-partialW);//a+b
			//VIP: we assume yTrain \in {-1,1}. Otherwise this error formula does not hold
			//check notebook October 12th 2011 for exact derivations of this error
			errorAux=b*b*partialW-2.0*b*cumSum[jj]+aPb*aPb*(totalW-partialW)-2.0*aPb*(totalCumSum-cumSum[jj]);
			
			//cout<<aPb-b<<" "<<b<<" "<<errorAux<<endl;				
			if(errorAux<errorMin)
			{
				bestWc.a=aPb-b;
				bestWc.b=b;
				bestWc.th=0.5*(xTrainPtr[idx]+xTrainPtr[sortIdxPtr[jj+1]]);
				bestWc.featureNdx=ii;
				errorMin=errorAux;
			}
			
		}				
	}
	delete[] sortIdxAux;
	delete[] intersection;
}
//--------------------------------------------------------------------------------------------------
//given a collection of samples find the best split 
//In order to be able to reuse the code recursively, we use listId to indicate the points belonging to this split
//numSamples is the total number of samples

feature fitTreeStumpToSingleFeature(const feature *xTrainPtr,const feature *yTrain,const feature *w,const unsigned long long int *sortIdxPtr,unsigned long long int numSamples,unsigned long long int numSamplesStump,const unsigned char *intersection,treeStump &bestWc,long long int featureNdx)
{
	feature errorAux,errorMin=1e32;
	feature aPb,b;
	feature *cumSum=new feature[numSamplesStump];
	feature *cumSumW=new feature[numSamplesStump];
	unsigned long long int *sortIdxAux=new unsigned long long int[numSamplesStump];
	/*
	 % [th, a , b] = fitRegressionStump(x, z);
	 % The regression has the form:
	 % z_hat = a * (x>th) + b;
	 %
	 % where (a,b,th) are so that it minimizes the weighted error:
	 % error = sum(w * |z - (a*(x>th) + b)|^2)
	 */
	//unsigned long long int *sortIdxPtr=&(sortIdx[ii*numSamples]);
	//feature *xTrainPtr=&(xTrain[ii*numSamples]);
	
	//find the subset of sortIdx that is meaningful for listId 
	if(numSamplesStump!=numSamples)//in the first pass we can avoid this part
	{
		unsigned long long int count=0;
		for(unsigned long long int jj=0;jj<numSamples && count<=numSamplesStump;jj++)
		{				
			if(intersection[sortIdxPtr[jj]]=='1') 
				sortIdxAux[count++]=sortIdxPtr[jj];
		}
		sortIdxPtr=sortIdxAux;
		if(count!=numSamplesStump)
		{
			cout<<"ERROR: mismatch between numSamplesStump "<<numSamplesStump<<" and counted samples "<<count<<endl;
			exit(2);
		}
	}
	//calculate optimal threshold
	unsigned long long int idx=sortIdxPtr[0];
	cumSum[0]=yTrain[idx]*w[idx];
	cumSumW[0]=w[idx];
	for(unsigned long long int jj=1;jj<numSamplesStump;jj++)
	{
		idx=sortIdxPtr[jj];
		cumSum[jj]=cumSum[jj-1]+yTrain[idx]*w[idx];
		cumSumW[jj]=cumSumW[jj-1]+w[idx];
	}
	
	feature partialW=0;
	feature totalCumSum=cumSum[numSamplesStump-1];
	feature totalW=cumSumW[numSamplesStump-1];
	
	for(unsigned long long int jj=0;jj<numSamplesStump-1;jj++)
	{
		idx=sortIdxPtr[jj];
		partialW=cumSumW[jj];
		b=cumSum[jj]/partialW;
		aPb=(totalCumSum-cumSum[jj])/(totalW-partialW);//a+b
		//VIP: we assume yTrain \in {-1,1}. Otherwise this error formula does not hold
		//check notebook October 12th 2011 for exact derivations of this error
		errorAux=b*b*partialW-2.0*b*cumSum[jj]+aPb*aPb*(totalW-partialW)-2.0*aPb*(totalCumSum-cumSum[jj]);
		
		//cout<<aPb-b<<" "<<b<<" "<<errorAux<<endl;				
		if(errorAux<errorMin)
		{
			bestWc.a=aPb-b;
			bestWc.b=b;
			bestWc.th=0.5*(xTrainPtr[idx]+xTrainPtr[sortIdxPtr[jj+1]]);
			bestWc.featureNdx=featureNdx;
			errorMin=errorAux;
		}
		
	}
	
	delete[] sortIdxAux;
	delete[] cumSum;
	delete[] cumSumW;
	return errorMin;
}

void fitTreeStumpParallelThreads(feature *xTrain,feature *yTrain,feature *w,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int *sortIdx,const vector<unsigned long long int> &listId,treeStump &bestWc)
{
	//specific case when we do not have any elements (to avoid breaking the code)
	if(listId.empty())
	{
		bestWc.a=0;
		bestWc.b=0;
		bestWc.th=0;
		bestWc.featureNdx=-1;
		bestWc.gtChild=-1;
		bestWc.leChild=-1;
		return;
	}
	//check for special case when all the values of yTrain are the same: the solution to the minimization is undefinded
	feature aux=yTrain[listId[0]];
	bool allEqual=true;
	for(vector<unsigned long long int>::const_iterator iter=listId.begin();iter!=listId.end();++iter)
	{
		if(fabs(aux-yTrain[*iter])>1e-3)
		{
			allEqual=false;
			break;
		}
	}
	if(allEqual)
	{
		//threshold is the maximum value of xTrain[feature=0]
		bestWc.th=-1e32;
		for(vector<unsigned long long int>::const_iterator iter=listId.begin();iter!=listId.end();++iter)
		{
			if(xTrain[*iter]>bestWc.th) 
				bestWc.th=xTrain[*iter];
		}
		bestWc.a=-aux-aux;
		bestWc.b=aux;
		bestWc.th+=1.0;//to give some margin
		bestWc.featureNdx=0;
		return;
	}
	
	
	unsigned long long int numSamplesStump=listId.size();
	unsigned char *intersection=new unsigned char[numSamples];
	memset(intersection,0,numSamples);
	for(vector<unsigned long long int>::const_iterator iter=listId.begin();iter!=listId.end();++iter) intersection[*iter]='1';//this is the same for all the features
	
	//select best regression stump
	feature *errorVec=new feature[numFeatures];
	vector<treeStump> bestWcVec(numFeatures);
	
	#pragma omp parallel for
	for(long long int ii=0;ii<(long long int)numFeatures;ii++)
	{
			
		/*
		 % [th, a , b] = fitRegressionStump(x, z);
		 % The regression has the form:
		 % z_hat = a * (x>th) + b;
		 %
		 % where (a,b,th) are so that it minimizes the weighted error:
		 % error = sum(w * |z - (a*(x>th) + b)|^2)
		 */
		unsigned long long int *sortIdxPtr=&(sortIdx[ii*numSamples]);
		feature *xTrainPtr=&(xTrain[ii*numSamples]);
		treeStump auxWc;
		errorVec[ii]=fitTreeStumpToSingleFeature(xTrainPtr,yTrain,w,sortIdxPtr,numSamples,numSamplesStump,intersection,auxWc,ii);
		bestWcVec[ii]=auxWc;
	}

	
	//find out best feature
	bestWc=bestWcVec[0];
	feature errorMin=errorVec[0];
	for(unsigned long long int ii=0;ii<numFeatures;ii++)
	{
		if(errorVec[ii]<errorMin)
		{
			bestWc=bestWcVec[ii];
			errorMin=errorVec[ii];
		}
	}
	
	delete[] errorVec;
	delete[] intersection;
}

//=================================================
//finds the median using parabolic interpolation
feature findL1center(feature *residTrain,unsigned long long int numSamples)
{
	feature a=0,b=0,c=0,x=0;
	feature aF,bF,cF,xF;//cost associated to each point
	//initial guess is the mean
	for(unsigned long long int ii=0;ii<numSamples;ii++) b+=residTrain[ii];
	
	b/=(feature)(numSamples);
	a=b-10.0;
	c=b+10.0;	
	
	aF=calculateLossFunctionL1(residTrain,numSamples,a);
	bF=calculateLossFunctionL1(residTrain,numSamples,b);
	cF=calculateLossFunctionL1(residTrain,numSamples,c);
	
	x=b;
	feature xOld=10*x;
	
	long long int numIter=0;
	while(fabs(x-xOld)/fabs(xOld)>0.1 && numIter<10)
	{
		xOld=x;
		x=b-0.5*((b-a)*(b-a)*(bF-cF)-(b-c)*(b-c)*(bF-aF))/((b-a)*(bF-cF)-(b-c)*(bF-aF));
		xF=calculateLossFunctionL1(residTrain,numSamples,x);
		if(x<a)
		{
			c=b;b=a;a=x;
			cF=bF;bF=aF;aF=xF;
		}else if(x<b){
			c=b;b=x;
			cF=bF;bF=xF;
		}else if(x<c){
			a=b;b=x;
			aF=bF;bF=xF;
		}else{
			a=b;b=c;c=x;
			aF=bF;bF=cF;cF=xF;
		}
		
		numIter++;
	}
	//cout<<"It took "<<numIter<<endl;
	return x;
}

//=================================================
//finds the Huber optimal location using parabolic interpolation
feature findHubercenter(feature *residTrain,unsigned long long int numSamples,feature delta)
{
	feature a=0,b=0,c=0,x=0;
	feature aF,bF,cF,xF;//cost associated to each point
	//initial guess is the mean
	for(unsigned long long int ii=0;ii<numSamples;ii++) b+=residTrain[ii];
	
	b/=(feature)(numSamples);
	a=b-10.0;
	c=b+10.0;	
	
	aF=calculateLossFunctionHuber(residTrain,numSamples,a,delta);
	bF=calculateLossFunctionHuber(residTrain,numSamples,b,delta);
	cF=calculateLossFunctionHuber(residTrain,numSamples,c,delta);
	
	x=b;
	feature xOld=10*x;
	
	long long int numIter=0;
	while(fabs(x-xOld)/fabs(xOld)>0.1 && numIter<10)
	{
		xOld=x;
		x=b-0.5*((b-a)*(b-a)*(bF-cF)-(b-c)*(b-c)*(bF-aF))/((b-a)*(bF-cF)-(b-c)*(bF-aF));
		xF=calculateLossFunctionHuber(residTrain,numSamples,x,delta);
		if(x<a)
		{
			c=b;b=a;a=x;
			cF=bF;bF=aF;aF=xF;
		}else if(x<b){
			c=b;b=x;
			cF=bF;bF=xF;
		}else if(x<c){
			a=b;b=x;
			aF=bF;bF=xF;
		}else{
			a=b;b=c;c=x;
			aF=bF;bF=cF;cF=xF;
		}
		
		numIter++;
	}
	//cout<<"It took "<<numIter<<endl;
	return x;
}

//===================================================================================================
//--------------------------------------for testing purposes----------------------------------------------
long long int mainTest(long long int argc, const char* argv[])
{
	
	if(argc!=6)
	{
		cout<<"ERROR: wrong number of arguments. Call has to be fileTrainingSet numSamples numFeatures numWeakClassifiers numLevelsPerTree"<<endl;
		return 1;
	}
	
	string filename(argv[1]);
	long long int numSamples=atoi(argv[2]);
	long long int numFeatures=atoi(argv[3]);
	long long int numWeakClassifiers=atoi(argv[4]);
	long long int J=atoi(argv[5]);
	time_t start,end;
	
	cout<<"Reading Num features="<<numFeatures<<";num samples="<<numSamples<<endl;
	
	ifstream fin(filename.c_str());
	
	//file format numSamples x numFeatures+1 matrix with blank space separation
	if(!fin.is_open())
	{
		cout<<"ERROR: file "<<filename<<" could not be opened"<<endl;
		return 1;
	}
	
	
	feature *xTrain=new feature[numSamples*numFeatures];
	feature *yTrain=new feature[numSamples];
	
	unsigned long long int pos=0;
	for(long long int ii=0;ii<numSamples;ii++)
	{
		pos=ii;
		for(long long int jj=0;jj<numFeatures;jj++)
		{
			fin>>xTrain[pos];
			pos+=numSamples;
		}
		fin>>yTrain[ii];
	}	
	fin.close();
	
	
	//------------------test incremental quantile---------------------------
	//------------------Numerical recipes incremental estimate is the slowest by far. qsort and parabolic method are on par-----------------------
	/*
	cout<<"Testing incremental quantile"<<endl;
	long long int numRuns2=1000;
	
	time(&start);
	for(long long int kk=0;kk<numRuns2;kk++)
	{
		IQagent medianAux;
		for(long long int ii=0;ii<numSamples*numFeatures;ii++)
		{
			medianAux.add(xTrain[ii]);
		}
	}
	time(&end);
	IQagent medianAux;
	for(long long int ii=0;ii<numSamples*numFeatures;ii++)
	{
		medianAux.add(xTrain[ii]);
	}
	cout<<"Estimated median is "<<medianAux.report(0.5)<<" in "<<difftime(end,start)/numRuns2<<" secs"<<endl;
	
	
	time(&start);
	for(long long int kk=0;kk<numRuns2;kk++)
		qsort(xTrain,numSamples*numFeatures,sizeof(feature),compare);
	time(&end);
	cout<<"Exact median is "<<xTrain[numSamples*numFeatures/2]<<" in "<<difftime(end,start)/numRuns2<<" secs"<<endl;
	
	
	time(&start);
	for(long long int kk=0;kk<numRuns2;kk++)
		findL1center(xTrain,numSamples*numFeatures);
	time(&end);
	
	cout<<"Gradient estimated median is "<<findL1center(xTrain,numSamples*numFeatures)<<" in "<<difftime(end,start)/numRuns2<<" secs"<<endl;
	
	return 0;
	 */
	//--------------------------------------------------------
	
	cout<<"Training classifier with "<<numWeakClassifiers<<" weak classifiers; Each weak classifiers is a tree with "<<J<<" levels"<<endl;
	vector< vector<treeStump> > classifier;
	time(&start);
	/*
	 //timing purposes
	 
	 long long int numRuns=10;
	 for(long long int ii=0;ii<numRuns;ii++)
	 gentleBoost(xTrain,yTrain, numWeakClassifiers, classifier);
	 
	 time(&end);
	 cout<<difftime(end,start)/numRuns<<"secs"<<endl;
	 */
	
	//cout<<"Using gentle boost"<<endl;
	//gentleTreeBoost(xTrain,yTrain,NULL, numWeakClassifiers, classifier,numSamples,numFeatures,J);
	
	cout<<"Using gradient boost with Huber loss function"<<endl;
	gradientTreeBoost(xTrain,yTrain, numWeakClassifiers, classifier,numSamples,numFeatures,J,Huber);
	
	time(&end);
	//---------------------------------------------------
	//write  out result
	string fileOut("tmpClassifier.txt");
	ofstream fout(fileOut.c_str());
	if(!fout.is_open())
	{
		cout<<"ERROR: file "<<fileOut<<" could not be opened"<<endl;
		return 1;
	}
	
	cout<<"Training took "<<difftime(end,start)<<"secs"<<endl;
	cout<<"Writing results in file "<<fileOut<<endl;
	for(unsigned long long int ii=0;ii<classifier.size();ii++)
	{
		fout<<"Classifier: "<<ii<<"----------------------------------"<<endl;
		for(unsigned long long int jj=0;jj<classifier[ii].size();jj++)
			//fout<<classifier[ii][jj].b<<" "<<classifier[ii][jj].featureNdx<<" "<<classifier[ii][jj].th<<" "<<classifier[ii][jj].a<<";"<<classifier[ii][jj].gtChild<<";"<<classifier[ii][jj].leChild<<";"<<endl;
			fout<<"Node "<<jj<<":"<<classifier[ii][jj].b<<" "<<classifier[ii][jj].featureNdx<<" "<<classifier[ii][jj].th<<" "<<classifier[ii][jj].a<<";"<<classifier[ii][jj].gtChild<<";"<<classifier[ii][jj].leChild<<";"<<endl;
	}
	
	
	fout.close();
	
	
	//----------------------------------------------------------------------------
	cout<<"Testing results from classifier"<<endl;
	boostingTreeClassifier(xTrain,yTrain,classifier,numSamples,numFeatures);
	for(long long int ii=0;ii<numSamples;ii++) cout<<yTrain[ii]<<" ";
	cout<<endl;
	
	//release memory
	delete []xTrain;
	delete []yTrain;
	
	return 0;
}

void transposeXtrainInPlace(feature *xTrain, unsigned long long int numSamples,unsigned long long int numFeatures)
{

	cout<<"ERROR: transposeXtrainInPlace: code not ready. For non-square matrices you need to study the cycles!!!!!!!"<<endl;
	exit(3);
	
	/*	
	feature aux;

	long long int posOrig =0;
	long long int posNew;
	for(unsigned long long int N = 0; N < numSamples; N++)
	{
		posNew = N;
		for(unsigned long long int p = 0; p < numFeatures; p++)
		{
			aux = xTrain[posOrig];
			xTrain[posOrig] = xTrain[posNew];
			xTrain[posNew] = aux;

			posNew += numSamples;
			posOrig++;
		}
	}
	*/
}

//=============================================================================================
void transposeXtrainOutOfPlace(feature *xTrain, unsigned long long int numSamples,unsigned long long int numFeatures)
{
	
	//copy memory
	long long int sizeX = ((long long int) (numSamples)) * ((long long int) (numFeatures));
	feature *xTrainAux = new feature[ sizeX ];
	memcpy(xTrainAux, xTrain, sizeof(feature) * sizeX);


	long long int posOrig;
	long long int posNew =0;	
	for(unsigned long long int p = 0; p < numFeatures; p++)
	{
		posOrig = p;
		for(unsigned long long int N = 0; N < numSamples; N++)	
		{
			xTrain[posNew] = xTrainAux[posOrig];

			posNew ++;
			posOrig += numFeatures;
		}
	}
	delete[] xTrainAux;
}


//----------------------------------------------------------------------------------------------------------------
//TODO: this function could be done faster by sorting Fx (since we would try every single threshold just making one change at a time, but it is not worth the effort)
void precisionRecallAccuracyCurve(feature* yTest, feature* Fx, unsigned long long int numSamples, ostream& out, feature thrStep)
{
	//feature thrStep = 0.3; 
	//find min and max of Fx to determine range of thresholds
	feature minFx = 1e32, maxFx = -1e32;
	for(long long int ii = 0;ii<numSamples; ii++)
	{
		minFx = min(Fx[ii],minFx);
		maxFx = max(Fx[ii],maxFx);
	}

	//calculate precision and recall for each thr level
	for(feature thr = minFx; thr < maxFx; thr+= thrStep) //arbitrary set of thresholds
	{
		long long int tp = 0, tn = 0, fp = 0, fn = 0;

		for(unsigned long long int jj = 0; jj<numSamples;jj++)
		{
			if( yTest[jj] >0 )//positive case
			{
				if( Fx[jj] > thr ) //predicted positive
					tp++;
				else
					fn++;
			}else{ //negative case
				if( Fx[jj] > thr ) //predicted positive
					fp++;
				else
					tn++;
			}
		}

		float accuracy = ((float) (tn + tp )) /( (float) (numSamples));
		float prec = ((float) (tp )) /( (float) (tp + fp));
		float recall = ((float) ( tp )) /( (float) (tp + fn));

		if( (tp + fp) >0 && (tp+fn) > 0) //threshold might be too large ->no more positives
			out<<thr<<","<<prec<<","<<recall<<","<<accuracy<<";"<<endl;//thr, precision, recall, accuracy
		else
			break;//we have reached to high of a threshold
	}
}

//======================================================================================================================
//From: [1] A. Niculescu-mizil and R. Caruana, “Obtaining Calibrated Probabilities from Boosting,” in In: Proc. 21st Conference on Uncertainty in Artificial Intelligence (UAI ’05), AUAI Press, 2005.
//Check Notebook January 30th 2013. P(y = 1 | Fx ) = 1 / (1 + exp( A*Fx + B ); We look for A,B with maximum likelihood
void calibrateBoostingScoreToProbabilityPlattMethod(feature* yTest, feature* Fx, unsigned long long int numSamples)
{
	//separate data into positive and negative samples
	vector<unsigned long long int> posIdx, negIdx;
	posIdx.reserve(numSamples / 2);
	negIdx.reserve(numSamples / 2);

	for(unsigned long long int ii = 0; ii< numSamples; ii++)
	{
		if( yTest[ii] > 0.5 )
			posIdx.push_back(ii);
		else
			negIdx.push_back(ii);
	}

	//grid search for A,B parameters
	double Astar = 0, Bstar = 0, fVal = 1e32, fValAux, aux;
	double Amin = -3.0, Amax = 3.0, Astep = 0.2;//change this to modify the grid search
	double Bmin = -3.0, Bmax = 3.0, Bstep = 0.2;//change this to modify the grid search
	for(double A = Amin; A < Amax; A += Astep)
	{
		for(double B = Bmin; B < Bmax; B += Bstep)
		{
			fValAux = 0.0;
			for(vector<unsigned long long int>::const_iterator iter = posIdx.begin(); iter!= posIdx.end(); ++iter)
			{
				fValAux += log( 1.0 + exp( A * Fx[*iter] + B ) );
			}
			for(vector<unsigned long long int>::const_iterator iter = negIdx.begin(); iter!= negIdx.end(); ++iter)
			{
				aux = A * Fx[*iter] + B;
				fValAux += log( 1.0 + exp( aux ) );
				fValAux -= aux; 
			}

			if (fValAux < fVal )
			{
				Astar = A;
				Bstar = B;
				fVal = fValAux;
			}
		}
	}

	cout<<"Optimal calibration for boosting scores into probabilities using Platt's Model  is A="<<Astar<<"; B="<<Bstar<<endl;
}
//===================================================================================================================================

long long int saveClassifier(const vector< vector<treeStump> > &classifier, string filename)
{
	if( classifier.empty() )
		return 0;

	ofstream out(filename.c_str() );
	if(!out.is_open() )
	{
		cout<<"ERROR: at saveClassifier: file "<<filename<<" could not be opened to save classifier"<<endl;
		return 1;
	}

	
	unsigned long long int numWeakClassifiers = classifier.size();
	
	out<<numWeakClassifiers<<endl;

	//feature thAux;
	for(unsigned long long int ii =0;ii<numWeakClassifiers; ii++)
	{
		out<<classifier[ii].size()<<endl;//number of stumps per weak classifier

		for(unsigned long long int jj = 0; jj < classifier[ii].size(); jj++)
		{
			out<<classifier[ii][jj].a <<" "<<classifier[ii][jj].b <<" "<<classifier[ii][jj].featureNdx <<" "<<classifier[ii][jj].gtChild <<" "<<classifier[ii][jj].leChild <<" "<<classifier[ii][jj].parentIdx <<" "<< classifier[ii][jj].th <<endl;
		}
	}
	
	out.close();
	return 0;
}
long long int loadClassifier(vector< vector<treeStump> > &classifier, string filename)
{
	
	ifstream out(filename.c_str() );
	if(!out.is_open() )
	{
		cout<<"ERROR: at loadClassifier: file "<<filename<<" could not be opened to load classifier"<<endl;
		return 1;
	}

	
	unsigned long long int numWeakClassifiers;
	
	out>>numWeakClassifiers;

	classifier.resize( numWeakClassifiers );
	for(unsigned long long int ii =0;ii<numWeakClassifiers; ii++)
	{
		unsigned long long int J;
		out>>J;
		classifier[ii].resize(J);
		for(unsigned long long int jj = 0; jj < classifier[ii].size(); jj++)
		{
			out>>classifier[ii][jj].a >>classifier[ii][jj].b >>classifier[ii][jj].featureNdx >>classifier[ii][jj].gtChild >>classifier[ii][jj].leChild >>classifier[ii][jj].parentIdx >>classifier[ii][jj].th;
		}
	}
	
	out.close();
	return 0;
}


long long int cleanTrainingSetTranspose(feature* xTrain, feature* yTrain, long long int *numSamples, long long int *nPos, long long int *nNeg, long long int numFeatures)
{
	vector<long long int> outlierIdx;//keeps location of outlier to move all the memory at the end in-place
	outlierIdx.reserve( (*numSamples) / 10);

	//calculate abs average for each feature
	vector<feature> avgVal(*numSamples);
	long long int count = 0;
	feature mean = 0, std = 0;
	for(long long int ii = 0; ii < *numSamples; ii++)
	{
		feature val = 0;
		for(long long int jj = 0; jj < numFeatures; jj++)
		{
			val += fabs(xTrain[count++]);
		}
		avgVal[ii] = val / ((feature) numFeatures );
		//calculate samples that seem outliers based on normality assumption
		mean += avgVal[ii];
		std += (avgVal[ii] * avgVal[ii]);
	}
	mean /= (( feature) (*numSamples) );
	std /= (( feature) (*numSamples) );
	std = sqrt( (std - mean * mean) * (( feature) (*numSamples) ) / (( feature) ((*numSamples) -1) ) );
	
	//detect outliers by assuming normality of the data
	feature K = 4.0;
	feature minThr = mean - K * std;
	feature maxThr = mean + K * std;
	for(long long int ii = 0; ii < *numSamples; ii++)
	{
		if( avgVal[ii] > maxThr || avgVal[ii] < minThr )
			outlierIdx.push_back(ii);
	}
	outlierIdx.push_back(-1);//to indicate end and avoid extra comparisons later

	//update xTrain, yTrain and numSamples by erasing in-place outliers
	if( outlierIdx.size() > 1)
	{
		long long int offsetNew = 0, offsetOrig = 0;
		long long int outlierIdxPos = 0;//we assume outlierIdx are sorted and we have added a -1 at the end to avoid an extra comparison
		for(long long int ii = 0; ii < *numSamples; ii++)
		{
			if( outlierIdx[ outlierIdxPos ] == ii )//this position is an outlier
			{
				outlierIdxPos++;//this is where the -1 at teh end is useful
				if( yTrain[ii] < 0 )
					(*nNeg)--;
				else
					(*nPos)--;
			}else {//this position is not an outlier

				if( offsetNew != offsetOrig)//copy set of features
				{
					memcpy(&(xTrain[offsetNew]), &(xTrain[offsetOrig]), sizeof(feature) * numFeatures);
					yTrain[ ii - outlierIdxPos] = yTrain[ii];
				}

				offsetNew += numFeatures;
			}
			offsetOrig += numFeatures;
		}
	}

	(*numSamples) -= (outlierIdx.size() - 1);


	//check
	if( *numSamples != ( (*nNeg) + (*nPos) ) )
	{
		cout<<"ERROR: at cleanTrainingSetTranspose: numSamples does not agree"<<endl;
		exit(3);
	}

	return ((long long int)(outlierIdx.size() -1) );
}
