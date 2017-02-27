#ifndef __ELLIPTICAL_HAAR_FEATURES_H__
#define __ELLIPTICAL_HAAR_FEATURES_H__

#include <math.h>
#include <fstream>


//parameters that should not change 
#ifndef CUDA_MAX_SIZE_CONST //to protect agains teh same constant define in other places in the code
#define CUDA_MAX_SIZE_CONST
static const int MAX_THREADS_CUDA = 1024; //adjust it for your GPU. This is correct for a 2.0 architecture
static const int MAX_BLOCKS_CUDA = 65535;
#endif

#ifndef DIMS_IMAGE_CONST //to protect agains teh same constant define in other places in the code
	#define DIMS_IMAGE_CONST
	static const int dimsImage = 3;//so thing can be set at co0mpile time
#endif

typedef unsigned short int imageType;//the kind sof images we are working with (you will need to recompile to work with other types)
static const int maxRadiusBox = 50; //maximum radius of the box around a Gaussian. To allow preallocation and avoid crashing with singular cases. IT SHOULD BE AT LEAST > DIAMETER OF LARGEST NUCLEI
static const int minRadiusBox = 2; //minimum radius of the box around a Gaussian. To allow preallocation and avoid crashing with singular cases

//for soem reason VS2010 complaints if I try to do float array[sizeMeanStdBoxVector]; even if it is a constant. So I need to precalculate it and CHANGE IT EVERYTIME I CHANGE MAX RADIUS BOX!!
//static const int sizeMeanStdBoxVector = (int)pow(2.0f,(int)ceil(log2((float)(maxDiameterBox*maxDiameterBox*maxDiameterBox)/((float)MAX_THREADS_CUDA))));//array with 2^n length to store partial statistics from the intensity of the box to normalize it
static const int sizeMeanStdBoxVector = 128;

//constant that depend on other constants and can be precalculated at compile time
static const int maxDiameterBox = 2 *maxRadiusBox + 1;


static const double r0 = 1.0;//initial radius (based on sigma) to define central cell ellipsoid
static const double kSigma = 1.0;//factor to define outer rings dimensions in the C-HoG block (r_m=(r0+m*kSigma)*sigma)
static const int numRadialCells = 3;//number of radial sectors in a block (including central)
static const int numAngCells = 0;//number of angular sectors in a block (central has only one) -we will use HEALPix to generate this -> 12*4^numAngCells 
static const double HEALPixLevel = 1;//number of possible orientations to bin gradient and create histogram (think in 3D -> so we are partitioning the sphere not the circle). We use HEALPix Matlab wrapper -> level=0->12bins (~60deg);level=1->48bins (~30deg);level=2->192bins (~15deg);

//constants that can be calculated from other constants
static const int numCellsHEALPix = (int)(12*pow(4.0,numAngCells)); // number of sectors per fixed radius using HEALpix
static const int numCellsIdx = 1 + 1 + numCellsHEALPix*(numRadialCells-1);//number of possible assigments for a boxel within a box;
static const int MAX_NUM_CELLS_IDX = 128; //to preallocate shared memory to caclulate average intensity in different cells 

//the typical access is as follows
/*
if(rr>=r0+(numRadialCells-1)*kSigma)
	cellIdx = 0;//voxel outside the considered ellipsoid
elseif(rr<r0)
	cellIdx=1; // center region (no angular sectors)
else
	for ss=1:numRadialCells-1
		for ll=1:(12*4^numAngCells)
			%pos=find(rr<r0+ss*kSigma & rr>=r0+(ss-1)*kSigma & idx==ll);
			cellIdx = 2 + ss*(12*4^numAngCells) + ll %angular sectors of the same radius are consecutive)
		end
	end
end
*/

/*
struct GaussianModelCUDA
{
    double   m[dimsImage];//center of the ellipsoid
    double   W[dimsImage*(dimsImage+1)/2];//symmetric matrix incidating covariance
};
*/

//contains basic feature vectors: averages for each cell and for each "onion layer" (ring)
//then we can use other functions to generate combinations of them
struct basicEllipticalHaarFeatureVector
{
	static bool useDoGfeatures;//true->we have twice as many features because of DoG (the deafault is true)
	static int numCells;
	static int numRings;
	float excentricity[dimsImage*(dimsImage-1)/2];//division between different eigenvalues
	float *ringAvgIntensity;//average intensity along every "onion layer" (includign teh center one)
	float *cellAvgIntensity;//average intensity in each angular sector at each radius value

	float *ringAvgIntensityDoG;//average intensity along every "onion layer" (includign teh center one)
	float *cellAvgIntensityDoG;//average intensity in each angular sector at each radius value

	basicEllipticalHaarFeatureVector()//constructor
	{
		numCells = (numRadialCells-1)*numCellsHEALPix;
		numRings = numRadialCells;
		ringAvgIntensity = new float[numRings];
		ringAvgIntensityDoG = new float[numRings];
		cellAvgIntensity = new float[numCells];
		cellAvgIntensityDoG = new float[numCells];
	}
	~basicEllipticalHaarFeatureVector()//destructor
	{
		delete[] ringAvgIntensity;
		delete[] ringAvgIntensityDoG;
		delete[] cellAvgIntensity;
		delete[] cellAvgIntensityDoG;
	}

	void basicEllipticalHaarFeatureVectorPrint(std::ostream& out)
	{
		for(int ii=0;ii<dimsImage;ii++)
			out<<excentricity[ii]<<", ";

		for(int ii=0;ii<numRings;ii++)
			out<<ringAvgIntensity[ii]<<", ";

		for(int ii=0;ii<numCells;ii++)
			out<<cellAvgIntensity[ii]<<",";

		for(int ii=0;ii<numRings;ii++)
			out<<ringAvgIntensityDoG[ii]<<",";

		for(int ii=0;ii<numCells-1;ii++)
			out<<cellAvgIntensityDoG[ii]<<",";
		
		out<<cellAvgIntensityDoG[numCells-1];
	}
};



//I only make visible functions without CUDA parameters on them, so I can add this header to any other file without CUDA dependencies

/*
\brief main routine to calculate features from image

m:				array of size numEllipsoids*dimsImage containing the mean for each Gaussian. It is ordered in a coalescent friendly manner for the GPU. So m=[m_x0,m_x1,....,m_y0,m_y1,....]
w:				array of size numEllipsoids*dimsImage*(dimsImage+1)/2 containing the covariance for each Gaussian. It is ordered in a coalescent friendly manner for the GPU.
numEllipsoids:	number of points to extract features from
im:				pointer to the image data.
dims:			dimsImage array indicating the dimensions of the image. The first dimension runs the fastest (we follow Mylib convention);
scaleIm:		dimsImage array indicating the scaling between different dimensions
devCUDA:		in case your machine has multiple GPU, to select which one you want to use.
symmetry:		integer (from 0 to 7) to allow computing symmetric features. Basically, given 3 bits, we decide which axis need to be flipped. This can be used during training in order to artifically extend the training set 8-fold.

returns:		float pointer of size numEllipsoids*numFeatures containing all the values for each feature and each point. Returns NULL if there was any error.
TODO: code only takes one possible image type right now.
*/
basicEllipticalHaarFeatureVector** calculateEllipticalHaarFeatures(const double *m,const double *W,int numEllipsoids,const imageType *im,const long long int *dims,int devCUDA,int symmetry=0);

/*
\brief	routine to extend the basic featurs calculated with calculateEllipticalHaarFeatures using subtration and addition. Check notebook on August 3rd 2012 for more details

fVec:		allocated float vector with numHaarFeaturesPerEllipsoid*numEllipsoids elements. For CPU cache friendly all the features for the ii-th ellipsoid are allocated from [ii*numHaarFeaturesPerEllipsoid,..,(ii+1)*numHaarFeaturesPerEllipsoid]
			If FVec=NULL -> program allocates memory
*/
void calculateCombinationsOfBasicHaarFeatures(basicEllipticalHaarFeatureVector** fBasic,int numEllipsoids,int *numHaarFeaturesPerEllipsoid, float** fVec_);

/*
\brief To be able to preallocate memory
*/
int getNumberOfHaarFeaturesPerllipsoid();

/*
\brief Calculate DoG in place in the GPU. Given a volume volCUDA (already allocated in CUDA memory) we calculate in place the separable convolution. There is a faster version in the SDK but this simple one should suffice

volCUDA:		pointer to memory in GPU containing the volume
volRadiusDims:	dimsImage array containing the radius of volCUDA along each dimension. So dimensions of volCUDA are 1+2*volRadiusDims
d:				dimsImage array containing the eigenvalues. We use them to calculate teh scale of teh Gaussian along each direction
*/
void calculateSeparableConvolutionBoxInCUDA(float* volCUDA,const int* volRadiusDims,const double* d);


#endif //__ELLIPTICAL_HAAR_FEATURES_H__