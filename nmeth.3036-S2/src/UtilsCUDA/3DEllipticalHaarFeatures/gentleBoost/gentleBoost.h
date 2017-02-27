#ifndef GENTLE_BOOST_H
#define GENTLE_BOOST_H

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//change this to define what kind of feature values are you using
//it has to be double if we want to use with Matlab:(
typedef float feature;
enum lossFunction {L_2,L_1,Huber};//TODO: L_1 results look awful. I need to debug that gradient boosting and see what is the problem
static const feature Huber_delta=0.5;//quantile to select the delta for Huber function. It has to be between [0 1.0]


using namespace std;

struct treeStump
{
	// Weak regression stump: It is defined by four parameters (a,b,k,th)
	// f_m = a * (x_k > th) + b
	long long int featureNdx;//selected feature
	feature th,a,b;
	long long int gtChild,leChild,parentIdx;//pointer (index) of the child we have to look at for trees with more than one level. If -1->it is a leave node
	
	treeStump & operator= (const treeStump & other)
	{
		if (this != &other) // protect against invalid self-assignment
        {
			featureNdx=other.featureNdx;
			th=other.th;
			a=other.a;
			b=other.b;
			gtChild=other.gtChild;
			leChild=other.leChild;
			parentIdx=other.parentIdx;
		}
		return *this;
	}
	
	treeStump()//default constructor
	{
		featureNdx=-1;
		th=a=b=0;
		gtChild=-1;
		leChild=-1;
		parentIdx=-1;
	}
	
	friend ostream& operator<<(ostream& output, const treeStump & p);
	friend bool operator==(const treeStump & p1, const treeStump & p2);
	friend bool operator!=(const treeStump & p1, const treeStump & p2);
};


//needed to sort values
struct weightAndSample
{
	feature xSample;
	unsigned long long int idx;
	
	weightAndSample()//default constructor
	{
		xSample=0;
		idx=0;
	}
};

bool compWeightAndSample(weightAndSample a, weightAndSample b);


//gentle boosting: based on Torralba's Matlab code which follows Hastie, Friedman and Tibishirani paper
//At each stage it weights the samples according to exponential loss (samples that are harder to predict get larger weights). Then it fits the best regression tree to the samples in the weighted least square sense
/*
xTrain is numSamples*numFeatures size array
i-th sample j-th feature ->xTrain[j*numSamples+i]
w represnets the initial weights for each sample. It can be set to NULL for equal weights. Very useful when positive samples are much smaller than negative samples.
*/
void gentleTreeBoost(feature *xTrain,feature *yTrain, feature *w,long long int numWeakClassifiers, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int J);


//Gradient tree boosting following Algorithm 10.3 from Hastie, Friedman and Tibishirani book "Elements of Statistical Learning"
//It uses different loss functions to measure residual between yTrain and prediction power so far. Then it fits the best regression tree to those residuals in the least square sense
/*
 xTrain is numSamples*numFeatures size array
 i-th sample j-th feature ->xTrain[j*numSamples+i]
 w represnets the initial weights for each sample. It can be set to NULL for equal weights. Very useful when positive samples are much smaller than negative samples.
 */
void gradientTreeBoost(feature *xTrain,feature *yTrain, long long int numWeakClassifiers, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int J,lossFunction L);


//returns regression value for a set of points. Seting the proper threshold is up to you
//works for gradientBoost and gentleBoost
void boostingTreeClassifier(feature *xTest,feature *Fx, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures);

//same as boostingTreeClassifier but xTest is assumed to be transposed 
void boostingTreeClassifierTranspose(feature *xTest,feature *Fx, vector< vector<treeStump> > &classifier,unsigned long long int numSamples,unsigned long long int numFeatures);


//saves classifier so it can be used later on
long long int saveClassifier(const vector< vector<treeStump> > &classifier, string filename);
long long int loadClassifier(vector< vector<treeStump> > &classifier, string filename);


//clean training datasets to avoid outliers. xTrain is tranposed with respect to gentleTreeBoost
//returns the number of outliers
long long int cleanTrainingSetTranspose(feature* xTrain, feature* yTrain, long long int *numSamples, long long int *nPos, long long int *nNeg, long long int numFeatures);

//------------------accuracy statistics (valid for any classifier) -----------------------------

void precisionRecallAccuracyCurve(feature* yTest, feature* Fx, unsigned long long int numSamples, ostream& out, feature thrStep = 0.3f);

void calibrateBoostingScoreToProbabilityPlattMethod(feature* yTest, feature* Fx, unsigned long long int numSamples);//so we can transform score into probability using sigmoid

//-----------------------------------------------------------------

//find optimal split for a colection of points using the least-squares loss function
//listId needs to be sorted!!
void fitTreeStump(feature *xTrain,feature *yTrain,feature *w,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int *sortIdx,const vector<unsigned long long int> &listId,treeStump &bestWc);
void fitTreeStumpParallelThreads(feature *xTrain,feature *yTrain,feature *w,unsigned long long int numSamples,unsigned long long int numFeatures,unsigned long long int *sortIdx,const vector<unsigned long long int> &listId,treeStump &bestWc);

//if for any reason you have xTrain as numFeatures x numSamples you need to transpose the matrix
void transposeXtrainInPlace(feature *xTrain, unsigned long long int numSamples,unsigned long long int numFeatures);
void transposeXtrainOutOfPlace(feature *xTrain, unsigned long long int numSamples,unsigned long long int numFeatures);


//-----------------support functions--------------------------
//find median (L_1 minimization) using parabolic interolation method
feature calculateLossFunctionL1(feature *residTrain,unsigned long long int numSamples,feature xHat);
feature findL1center(feature *residTrain,unsigned long long int numSamples);
//find Huber (L_1 minimization) using parabolic interolation method
feature calculateLossFunctionHuber(feature *residTrain,unsigned long long int numSamples,feature xHat,feature delta);
feature findHubercenter(feature *residTrain,unsigned long long int numSamples,feature delta);


inline feature calculateLossFunctionL1(feature *residTrain,unsigned long long int numSamples,feature xHat)
{
	feature c=0;
	for(unsigned long long int ii=0;ii<numSamples;ii++) c+=fabs(xHat-residTrain[ii]);
	return c;
}
inline feature calculateLossFunctionHuber(feature *residTrain,unsigned long long int numSamples,feature xHat,feature delta)
{
	feature c=0,aux;
	for(unsigned long long int ii=0;ii<numSamples;ii++)
	{
		aux=xHat-residTrain[ii];
		if(fabs(aux)<delta) c+=(aux*aux);
		else  c+=delta*fabs(aux);

	}
	return c;
}

#endif GENTLE_BOOST_H