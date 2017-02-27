/*
 * GMEMupdateCUDA.cu
 *
 *  Created on: Jul 15, 2011
 *      Author: amatf
 */

//#define USE_CUDA_PRINTF

#include "GMEMcommonCUDA.h"
#include "GMEMupdateCUDA.h"
#include "knnCuda.h"
#include "external/book.h"
#include <algorithm>
#include <math.h>
#include <iostream>
#include <fstream>

#ifdef USE_CUDA_PRINTF
#include "external/cuPrintf.cu"
#endif


#if defined(_WIN32) || defined(_WIN64)
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif



#include "cusparse.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "../constants.h"

using namespace std;

__constant__ float scaleGMEMCUDA[dimsImage];

static unsigned int iterEMdebug=0;//used to print out each of the EM iterations

//neded for digamma function
__device__ static const double C = 12.0;
__device__ static const double D1 = -0.57721566490153286;
__device__ static const double D2 = 1.6449340668482264365;
__device__ static const double S = 1e-6;
__device__ static const double S3 = 1.0 / 12.0;
__device__ static const double S4 = 1.0 / 120.0;
__device__ static const double S5 = 1.0 / 252.0;
__device__ static const double S6 = 1.0 / 240.0;
__device__ static const double S7 = 1.0 / 132.0;

//needed for other functions
__device__ static const double expectedLogC1=dimsImage*0.69314718055994528623;//dimsImage*log(2.0);
__device__ static const double pow_2Pi_dimsImage2=15.74960994572241901324;//(2*pi)^(dimsImage/2.0) For Gaussian evaluation
__device__ static const double pow_2Pi_dimsImage=248.05021344239852965075;//(2*pi)^(dimsImage) For Gaussian evaluation

__device__ static const double PI_=3.14159265358979311600;
__device__ static const double SQRT3_=1.73205080756887719318;


//=========================================================================================
__device__ GaussianMixtureModelCUDA& GaussianMixtureModelCUDA::operator=(const GaussianMixtureModelCUDA& p)
{	
	if (this != &p)
	{
		int count=0;
		for(int ii=0;ii<dimsImage;ii++)
		{
			m_k[ii]=p.m_k[ii];
			m_o[ii]=p.m_o[ii];
			for(int jj=ii;jj<dimsImage;jj++)
			{
				W_k[count]=p.W_k[count];
				W_o[count]=p.W_o[count];
				count++;
			}
		}
		
		beta_k=p.beta_k;		
		nu_k=p.nu_k;
		alpha_k=p.alpha_k;
	
		beta_o=p.beta_o;
		nu_o=p.nu_o;
		alpha_o=p.alpha_o;
		
		splitScore=p.splitScore;
		fixed=p.fixed;

		expectedLogDetCovarianceCUDA=p.expectedLogDetCovarianceCUDA;
		expectedLogResponsivityCUDA=p.expectedLogResponsivityCUDA;

		//CUDA  does not allowed to call std::vector functions inside __device__ function because it requires memory creation destruction. But I do not need this in here
		//supervoxelIdx = p.supervoxelIdx;
		//for(int ii =0;ii<MAX_SUPERVOXELS_PER_GAUSSIAN;ii++)
		//	supervoxelIdx[ii] = p.supervoxelIdx[ii]; 
	}
	return *this;
}
//============================================================================================
//we assume dimsImage=3 to speed up the code by loop unrolling
__device__ inline double expectedMahalanobisDistanceCUDA(GaussianMixtureModelCUDA *p, float *x_n )
{
	double dx[dimsImage];

	dx[0]=x_n[0]-p->m_k[0];dx[1]=x_n[1]-p->m_k[1];dx[2]=x_n[2]-p->m_k[2];

	return ((double)dimsImage)/p->beta_k+p->nu_k*(p->W_k[0]*dx[0]*dx[0]+p->W_k[3]*dx[1]*dx[1]+p->W_k[5]*dx[2]*dx[2]+2.0*p->W_k[1]*dx[0]*dx[1]+2.0*p->W_k[2]*dx[0]*dx[2]+2.0*p->W_k[4]*dx[1]*dx[2]);
	//return (dimsImage/beta_k+nu_k*(x_n-m_k).transpose()*W_k*(x_n-m_k));
}
//we assume dimsImage=3 to speed up the code by loop unrolling
__device__ inline double mahalanobisDistanceCUDA(GaussianMixtureModelCUDA *p, float *x_n )
{
	double dx[dimsImage];

	dx[0]=x_n[0]-p->m_k[0];dx[1]=x_n[1]-p->m_k[1];dx[2]=x_n[2]-p->m_k[2];

	return p->nu_k*(p->W_k[0]*dx[0]*dx[0]+p->W_k[3]*dx[1]*dx[1]+p->W_k[5]*dx[2]*dx[2]+2.0*p->W_k[1]*dx[0]*dx[1]+2.0*p->W_k[2]*dx[0]*dx[2]+2.0*p->W_k[4]*dx[1]*dx[2]);
	//return (dimsImage/beta_k+nu_k*(x_n-m_k).transpose()*W_k*(x_n-m_k));
}

//===========================================================================
//From CUDA_Programming_GUIDE section B.11
__device__ inline double atomicAddCUDA(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val +__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}
//===========================================================================

//From CUDA_Programming_GUIDE section B.11
__device__ inline float atomicAddCUDA(float* address, float val)
{
	unsigned  int* address_as_ull = (unsigned int*)address;
	unsigned  int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__float_as_int(val +__int_as_float(assumed)));
	} while (assumed != old);
	return __int_as_float(old);
}

//=========================================================================
//determinant for 3x3 symmetric matrix
__device__ inline double determinantSymmetricW_3D(const double *W_k)
{
	return W_k[0]*(W_k[3]*W_k[5]-W_k[4]*W_k[4])-W_k[1]*(W_k[1]*W_k[5]-W_k[2]*W_k[4])+W_k[2]*(W_k[1]*W_k[4]-W_k[2]*W_k[3]);
}
//=========================================================================
//inverse for a 3x3 symmetric matrix
__device__ inline void inverseSymmetricW_3D(double *W,double *W_inverse)
{
	double detW=determinantSymmetricW_3D(W);
	if(fabs(detW)<1e-16) //matrix is singular
	{
		W_inverse[0]=1e300;W_inverse[1]=1e300;W_inverse[2]=1e300;W_inverse[3]=1e300;W_inverse[4]=1e300;W_inverse[5]=1e300;
		return;
	}
	W_inverse[0]=(W[3]*W[5]-W[4]*W[4])/detW;
	W_inverse[1]=(W[4]*W[2]-W[1]*W[5])/detW;
	W_inverse[2]=(W[1]*W[4]-W[3]*W[2])/detW;


	W_inverse[3]=(W[0]*W[5]-W[2]*W[2])/detW;
	W_inverse[4]=(W[1]*W[2]-W[0]*W[4])/detW;

	W_inverse[5]=(W[0]*W[3]-W[1]*W[1])/detW;

	return;
}

//===========================================================================
//analytical solution for eigenvalues 3x3 real symmetric matrices
//formula for eigenvalues from http://en.wikipedia.org/wiki/Eigenvalue_algorithm#Eigenvalues_of_3.C3.973_matrices
__device__ inline void  eig3(const double *w, double *d, double *v,int vIsZero2)
{

	double m,p,q;
	int vIsZero=0;
	double phi,aux1,aux2,aux3;


	//calculate determinant to check if matrix is singular
	q=determinantSymmetricW_3D(w);

	if(fabs(q)<1e-24)//we consider matrix is singular
	{
		d[0]=0.0;
		//solve a quadratic equation
		m=-w[0]-w[3]-w[5];
		q=-w[1]*w[1]-w[2]*w[2]-w[4]*w[4]+w[0]*w[3]+w[0]*w[5]+w[3]*w[5];
		p=m*m-4.0*q;
		if(p<0) p=0.0;//to avoid numerical errors (symmetric matrix should have real eigenvalues)
		else p=sqrt(p);
		d[1]=0.5*(-m+p);
		d[2]=0.5*(-m-p);

	}else{//matrix not singular
		m=(w[0]+w[3]+w[5])/3.0;//trace of w /3
		q=0.5*((w[0]-m)*((w[3]-m)*(w[5]-m)-w[4]*w[4])-w[1]*(w[1]*(w[5]-m)-w[2]*w[4])+w[2]*(w[1]*w[4]-w[2]*(w[3]-m)));//determinant(a-mI)/2
		p=(2.0*(w[1]*w[1]+w[2]*w[2]+w[4]*w[4])+(w[0]-m)*(w[0]-m)+(w[3]-m)*(w[3]-m)+(w[5]-m)*(w[5]-m))/6.0;


		//NOTE: the follow formula assume accurate computation and therefor q/p^(3/2) should be in range of [1,-1],
		//but in real code, because of numerical errors, it must be checked. Thus, in case abs(q) >= abs(p^(3/2)), set phi = 0;
		phi= q / pow(p,1.5);
		if(phi <= -1)
			phi = PI_ / 3.0;
		else if (phi >= 1)
			phi = 0;
		else 
			phi = acos(phi)/3.0;

		aux1=cos(phi);aux2=sin(phi);aux3=sqrt(p);

		//eigenvalues
		d[0] = m + 2.0*aux3*aux1;
		d[1] = m - aux3*(aux1 + SQRT3_*aux2);
		d[2] = m - aux3*(aux1 - SQRT3_*aux2);
	}

	//eigenvectors
	v[0]=w[1]*w[4]-w[2]*(w[3]-d[0]); v[1]=w[2]*w[1]-w[4]*(w[0]-d[0]); v[2]=(w[0]-d[0])*(w[3]-d[0])-w[1]*w[1];
	v[3]=w[1]*w[4]-w[2]*(w[3]-d[1]); v[4]=w[2]*w[1]-w[4]*(w[0]-d[1]); v[5]=(w[0]-d[1])*(w[3]-d[1])-w[1]*w[1];
	v[6]=w[1]*w[4]-w[2]*(w[3]-d[2]); v[7]=w[2]*w[1]-w[4]*(w[0]-d[2]); v[8]=(w[0]-d[2])*(w[3]-d[2])-w[1]*w[1];

	/*
	if(vIsZero2==91)
	{
		printf("--------------------Inside kernel--------------\n");
		printf("norm1=%g\n",v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
		printf("norm2=%g\n",v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);
		printf("norm3=%g\n",v[6]*v[6]+v[7]*v[7]+v[8]*v[8]);
		printf("eigenVector1(unnormalized)=%g %g %g\n",v[0],v[1],v[2]);
		printf("eigenVector2(unnormalized)=%g %g %g\n",v[3],v[4],v[5]);
		printf("eigenVector3(unnormalized)=%g %g %g\n",v[6],v[7],v[8]);
		printf("--------Kernel finished--------\n");
	}
   */


	//normalize eigenvectors
	phi=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
	if(phi>1e-12){ v[0]/=phi;v[1]/=phi;v[2]/=phi;}
	else{//numerically seems zero: we need to try the other pair of vectors to form the null space (it could be that v1 and v2 were parallel)
		v[0]=w[1]*(w[5]-d[0])-w[2]*w[4];v[1]=w[2]*w[2]-(w[5]-d[0])*(w[0]-d[0]);v[2]=(w[0]-d[0])*w[4]-w[1]*w[2];
		phi=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
		if(phi>1e-12){v[0]/=phi;v[1]/=phi;v[2]/=phi;}
		else vIsZero+=1;
	}    


	phi=sqrt(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);
	if(phi>1e-12){ v[3]/=phi;v[4]/=phi;v[5]/=phi;}
	else{//numerically seems zero: we need to try the 
		v[3]=w[1]*(w[5]-d[1])-w[2]*w[4];v[4]=w[2]*w[2]-(w[5]-d[1])*(w[0]-d[1]);v[5]=(w[0]-d[1])*w[4]-w[1]*w[2];
		phi=sqrt(v[3]*v[3]+v[4]*v[4]+v[5]*v[5]);
		if(phi>1e-12){v[3]/=phi;v[4]/=phi;v[5]/=phi;}
		else vIsZero+=2;
	}

	phi=sqrt(v[6]*v[6]+v[7]*v[7]+v[8]*v[8]);
	if(phi>1e-12) {v[6]/=phi;v[7]/=phi;v[8]/=phi;}
	else{//numerically seems zero: we need to try the 
		v[6]=w[1]*(w[5]-d[2])-w[2]*w[4];v[7]=w[2]*w[2]-(w[5]-d[2])*(w[0]-d[2]);v[8]=(w[0]-d[2])*w[4]-w[1]*w[2];
		phi=sqrt(v[6]*v[6]+v[7]*v[7]+v[8]*v[8]);
		if(phi>1e-12){v[6]/=phi;v[7]/=phi;v[8]/=phi;}
		else vIsZero+=4;
	}

	//adjust v in case some eigenvalues are zeros
	switch(vIsZero)
	{
	case 1:
		v[0]=v[4]*v[8]-v[5]*v[7];
		v[1]=v[5]*v[6]-v[3]*v[8];
		v[2]=v[4]*v[6]-v[3]*v[7];
		break;

	case 2:
		v[3]=v[1]*v[8]-v[2]*v[7];
		v[4]=v[2]*v[6]-v[0]*v[8];
		v[5]=v[1]*v[6]-v[0]*v[7];
		break;

	case 4:
		v[6]=v[4]*v[2]-v[5]*v[1];
		v[7]=v[5]*v[0]-v[3]*v[2];
		v[8]=v[4]*v[0]-v[3]*v[1];
		break;
	case 3:
		phi=sqrt(v[7]*v[7]+v[6]*v[6]);
		if(phi<1e-12)//it means first eigenvector is [0 0 1]
		                                              {v[3]=1.0;v[4]=0.0;v[5]=0.0;}
		else{ v[3]=-v[7]/phi;v[4]=v[6]/phi;v[5]=0.0;}
		v[0]=v[4]*v[8]-v[5]*v[7];
		v[1]=v[5]*v[6]-v[3]*v[8];
		v[2]=v[3]*v[7]-v[4]*v[6];
		break;

	case 6:
		phi=sqrt(v[1]*v[1]+v[0]*v[0]);
		if(phi<1e-12)//it means first eigenvector is [0 0 1]
		{v[6]=1.0;v[7]=0.0;v[8]=0.0;}
		else{ v[6]=-v[1]/phi;v[7]=v[0]/phi;v[8]=0.0;}
		v[3]=v[1]*v[8]-v[2]*v[7];
		v[4]=v[2]*v[6]-v[0]*v[8];
		v[5]=v[0]*v[7]-v[1]*v[6];
		break;

	case 5:
		phi=sqrt(v[4]*v[4]+v[5]*v[5]);
		if(phi<1e-12)//it means first eigenvector is [0 0 1]
		{v[0]=1.0;v[1]=0.0;v[2]=0.0;}
		else{ v[0]=-v[4]/phi;v[1]=v[5]/phi;v[2]=0.0;}
		v[6]=v[4]*v[2]-v[5]*v[1];
		v[7]=v[5]*v[0]-v[3]*v[2];
		v[8]=v[1]*v[3]-v[4]*v[0];
		break;

	case 7://matrix is basically zero: so we set eigenvectors to identity matrix
		v[1]=v[2]=v[3]=v[5]=v[6]=v[7]=0.0;
		v[0]=v[4]=v[8]=1.0;
		break;

	}

	//make sure determinant is +1 for teh rotation matrix
	phi=v[0]*(v[4]*v[8]-v[5]*v[7])-v[1]*(v[3]*v[8]-v[5]*v[6])+v[2]*(v[3]*v[7]-v[4]*v[6]);
	if(phi<0)
	{
		v[0]=-v[0];v[1]=-v[1];v[2]=-v[2];
	}	
}
//----------------------------------------------------------------
//-------------------------------------------------------------------
//analytical solution for eigenvalues 2x2 real symmetric matrices
__device__ inline void eig2(const double *w, double *d, double *v)
{
	double aux1,phi;
	int vIsZero=0;

	aux1=(w[0]+w[2])/2.0;
	phi=sqrt(4.0*w[1]*w[1] + (w[0]-w[2])*(w[0]-w[2]))/2.0;

	d[0] = aux1 + phi;
	d[1] = aux1 - phi;


	//calculate eigenvectors
	//eigenvectors
	v[0]=-w[1];v[1]=w[0]-d[0];
	v[2]=-w[1];v[3]=w[0]-d[1];

	//normalize eigenvectors
	phi=sqrt(v[0]*v[0]+v[1]*v[1]);
	if(phi>0){ v[0]/=phi;v[1]/=phi;}
	else vIsZero+=1;

	phi=sqrt(v[2]*v[2]+v[3]*v[3]);
	if(phi>0){ v[2]/=phi;v[3]/=phi;}
	else vIsZero+=2;

	switch(vIsZero)
	{
	case 1:
		v[0]=-v[3];v[1]=v[2];
		break;
	case 2:
		v[2]=-v[1];v[3]=v[0];
		break;
	case 3://matrix is basically zero: so we set eigenvectors to identity matrix
		v[1]=v[2]=0.0;
		v[0]=v[3]=1.0;
		break;
	}
	//make sure determinant is +1 for teh rotation matrix
	phi=v[0]*v[3]-v[1]*v[2];
	if(phi<0)
	{
		v[0]=-v[0];v[1]=-v[1];
	}
}
//==========================================================================
/// Computes the Digamma function which is mathematically defined as the derivative of the logarithm of the gamma function.
/// This implementation is based on
///     Jose Bernardo
///     Algorithm AS 103:
///     Psi ( Digamma ) Function,
///     Applied Statistics,
///     Volume 25, Number 3, 1976, pages 315-317.
/// Using the modifications as in Tom Minka's lightspeed toolbox.
/// </summary>
/// <param name="x">The argument of the digamma function.</param>
/// <returns>The value of the DiGamma function at <paramref name="x"/>.</returns>
__device__ __host__ inline double DiGammaPositiveX(double x)
{
	if (x <= S)
	{
		return D1 - (1 / x) + (D2 * x);
	}

	double result = 0;
	while (x < C)
	{
		result -= 1 / x;
		x++;
	}

	if (x >= C)
	{
		double r = 1.0 / x;
		result += log(x) - (0.5 * r);
		r *= r;

		result -= r * (S3 - (r * (S4 - (r * (S5 - (r * (S6 - (r * S7))))))));
	}

	return result;
}
__device__ __host__ inline double DiGamma(double x)
{
	// Handle special cases.
	if (x <= 0 && floor(x) == x)
	{
		return -1e300;//negative infinite
	}
	// Use inversion formula for negative numbers.
	if (x < 0)
	{
		return DiGammaPositiveX(1.0 - x) + (PI_CUDA / tan(-PI_CUDA * x));
	}else return DiGammaPositiveX(x);
}
//============================================================================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMcopyGaussianCenter2ConstantMemoryKernel(GaussianMixtureModelCUDA *vecGMCUDA,float *refTempCUDA,int ref_nb)
		{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<ref_nb)
	{
		GaussianMixtureModelCUDA *GM=&(vecGMCUDA[tid]);
		tid*=dimsImage;
		//loop unroll
		refTempCUDA[tid++]=GM->m_k[0];//non-ocallescent access
		refTempCUDA[tid++]=GM->m_k[1];
		refTempCUDA[tid++]=GM->m_k[2];
	}
		}

//============================================================================================================================
__global__ void __launch_bounds__(MAX_THREADS) memCpyDeviceToDeviceKernel(int *src,int *dest,long long int numElem)
		{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while(tid<numElem)
	{
		dest[tid]=src[tid];
		tid+=offset;
	}
}
//=====================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMexpectedLogDetCovarianceKernel(GaussianMixtureModelCUDA *vecGMCUDA,int ref_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	double out,aux;
	GaussianMixtureModelCUDA *GM;

	if(tid<ref_nb)//we might launch more kernels than needed
	{
		GM=&(vecGMCUDA[tid]);
		if(GM->m_o[0]<-1e31)//dead cell
		{
			GM->expectedLogDetCovarianceCUDA=-1e32;
		}else{
			aux=GM->nu_k+1.0;
			out=expectedLogC1+log(determinantSymmetricW_3D(GM->W_k));
			for(int ii=0;ii<dimsImage;ii++)
			{
				out+=DiGamma(0.5*(aux-ii));
			}
			GM->expectedLogDetCovarianceCUDA=out;
		}
	}
				}
//=====================================================================
//kernel to calculate totalAlpha: based on dot product exmaple
__global__ void __launch_bounds__(MAX_THREADS) GMEMtotalAlphaKernel(GaussianMixtureModelCUDA *vecGMCUDA,double *totalAlphaTempCUDA,int ref_nb)
				{
	__shared__ double cache[MAX_THREADS];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	// set the cache values
	if(tid<ref_nb) cache[cacheIndex] = vecGMCUDA[tid].alpha_k;
	else cache[cacheIndex] = 0.0;
	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		totalAlphaTempCUDA[blockIdx.x] = cache[0];
				}

//=====================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMexpectedLogResponsivityKernel(GaussianMixtureModelCUDA *vecGMCUDA,int ref_nb,double totalAlphaDiGamma)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//calculate log responsivity
	if(tid<ref_nb)//we might launch more kernels than needed
	{
		GaussianMixtureModelCUDA *GM=&(vecGMCUDA[tid]);
		if(GM->m_o[0]<-1e31)//dead cell
		{
			GM->expectedLogResponsivityCUDA=-1e32;
		}else GM->expectedLogResponsivityCUDA=DiGamma(GM->alpha_k)-totalAlphaDiGamma;

#ifdef USE_CUDA_PRINTF
		//cuPrintf("alpha_k=%g;DiGamma=%g;totalDiGamma=%g\n",GM->alpha_k,DiGamma(GM->alpha_k),totalAlphaDiGamma);
#endif
	}
				}
//=======================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeXkNkKernel(int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *X_k,float *N_k,long long int query_nb,int ref_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	long long int pos;
	float x_n[dimsImage];
	//float imgData;  we already multiplied while computing rnk
	float aux;
	int idxAux;

	while(tid<query_nb)
	{
		pos=tid;
		//loop unrolling for 3D
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			idxAux=indCUDA[pos];//coalescent access
			aux=(float)rnkCUDA[pos];//coalescence access
			if(aux>1e-3)//to avoid unnecessary atomicAdd accesses: rnk has been normalized, so it should be high
			{
				//aux*=imgData;

				atomicAddCUDA(&(N_k[idxAux]),aux);
				//loop unroll
				atomicAddCUDA(&(X_k[idxAux]),aux*x_n[0]);
				idxAux+=ref_nb;
				atomicAddCUDA(&(X_k[idxAux]),aux*x_n[1]);
				idxAux+=ref_nb;
				atomicAddCUDA(&(X_k[idxAux]),aux*x_n[2]);
			}			
			pos+=query_nb;
		}

		//update pointer for next query_point to check
		tid+=offset;
	}
				}
//=======================================================================
//this kernel has to be launch with MAX_THREADS (and with as large of a number as possible)
__global__ void __launch_bounds__(MAX_THREADS_CUDA) GMEMcomputeXkNkKernelNoAtomic(int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *X_k,float *N_k,long long int query_nb,int ref_nb)
				{

	__shared__ float auxX_k0[MAX_THREADS_CUDA];
	__shared__ float auxX_k1[MAX_THREADS_CUDA];
	__shared__ float auxX_k2[MAX_THREADS_CUDA];
	__shared__ float auxN_k[MAX_THREADS_CUDA];
	//reset counter
	auxX_k0[threadIdx.x]=0.0f;
	auxX_k1[threadIdx.x]=0.0f;
	auxX_k2[threadIdx.x]=0.0f;
	auxN_k[threadIdx.x]=0.0f;
	__syncthreads();

	float aux;
	long long int pos;

	for(long long int ii=threadIdx.x;ii<query_nb*maxGaussiansPerVoxel;ii+=MAX_THREADS_CUDA)//to have colescent access indCUDA
	{
		if(indCUDA[ii]==blockIdx.x)//each block of threads processes a specific Gaussian
		{
			aux=(float)rnkCUDA[ii];
			if(aux>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
			{
				auxN_k[threadIdx.x]+=aux;
				pos=ii%query_nb;
				auxX_k0[threadIdx.x]+=aux*queryCUDA[pos];
				pos+=query_nb;
				auxX_k1[threadIdx.x]+=aux*queryCUDA[pos];
				pos+=query_nb;
				auxX_k2[threadIdx.x]+=aux*queryCUDA[pos];
			}
		}
	}

	__syncthreads();

	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i) auxN_k[cacheIndex] += auxN_k[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) N_k[blockIdx.x] = auxN_k[0];
	__syncthreads();
	//----------------------------------
	aux=auxN_k[0];
	if(aux>0.0f)
	{
		pos=blockIdx.x;
		i = blockDim.x/2;
		while (i != 0) 
		{
			if (cacheIndex < i) auxX_k0[cacheIndex] += auxX_k0[cacheIndex + i];
			__syncthreads();
			i /= 2;
		}
		if (cacheIndex == 0) X_k[pos] = auxX_k0[0]/aux;//we don't need normalization later on!!
		__syncthreads();

		//----------------------------------
		pos+=ref_nb;
		i = blockDim.x/2;
		while (i != 0) 
		{
			if (cacheIndex < i) auxX_k1[cacheIndex] += auxX_k1[cacheIndex + i];
			__syncthreads();
			i /= 2;
		}
		if (cacheIndex == 0) X_k[pos] = auxX_k1[0]/aux;
		__syncthreads();

		//----------------------------------
		pos+=ref_nb;
		i = blockDim.x/2;
		while (i != 0) 
		{
			if (cacheIndex < i) auxX_k2[cacheIndex] += auxX_k2[cacheIndex + i];
			__syncthreads();
			i /= 2;
		}
		if (cacheIndex == 0) X_k[pos] = auxX_k2[0]/aux;
	}
}

//=======================================================================
//this kernel has to be launch with MAX_THREADS (and with as large of a number as possible)
__global__ void __launch_bounds__(MAX_THREADS_CUDA) GMEMcalculateLocalKullbackDiversityKernel(int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *N_k,GaussianMixtureModelCUDA *vecGMCUDA,long long int query_nb,int ref_nb)
				{


	__shared__ float auxJ_k[MAX_THREADS_CUDA];
	//reset counter
	auxJ_k[threadIdx.x]=0.0f;
	__syncthreads();

	double aux;
	long long int pos;
	float x_n[dimsImage];


	GaussianMixtureModelCUDA *GM=&(vecGMCUDA[blockIdx.x]);
	double logZp=log(sqrt(determinantSymmetricW_3D(GM->W_k)*pow(GM->nu_k,dimsImage))/pow_2Pi_dimsImage2);

	for(long long int ii=threadIdx.x;ii<query_nb*maxGaussiansPerVoxel;ii+=MAX_THREADS_CUDA)//to have colescent access indCUDA
	{
		if(indCUDA[ii]==blockIdx.x)//each block of threads processes a specific Gaussian
		{
			aux=(float)rnkCUDA[ii];
			if(aux>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
			{

				pos=ii%query_nb;
				x_n[0]=queryCUDA[pos];
				pos+=query_nb;
				x_n[1]=queryCUDA[pos];
				pos+=query_nb;
				x_n[2]=queryCUDA[pos];

				auxJ_k[threadIdx.x]+=aux*(log(aux)+0.5*mahalanobisDistanceCUDA(GM, x_n )-logZp);

			}
		}
	}

	__syncthreads();

	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i) auxJ_k[cacheIndex] += auxJ_k[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	aux=N_k[blockIdx.x];
	if (cacheIndex == 0 && aux>0.0) GM->splitScore = (auxJ_k[0]/aux)-log(aux);

				}
//=======================================================================
//this thread needs to be launched with gridDim.x=ref_nb and MAX_THREADS_CUDA
__global__ void __launch_bounds__(MAX_THREADS_CUDA) GMEMcalculateLocalKullbackDiversityKernelTr( float *queryCUDA,float2 *rnkIndCUDA,float *N_k,int *csrPtrRowACUDA,GaussianMixtureModelCUDA *vecGMCUDA,long long int query_nb,int ref_nb)
		{


	__shared__ float auxJ_k[MAX_THREADS_CUDA];
	__shared__ int pIni;
	__shared__ int pEnd;

	GaussianMixtureModelCUDA *GM=&(vecGMCUDA[blockIdx.x]);
	//reset counter
	auxJ_k[threadIdx.x]=0.0f;
	if(threadIdx.x==0)//check that we have elements to count
	{
		pIni=csrPtrRowACUDA[blockIdx.x];
		pEnd=csrPtrRowACUDA[blockIdx.x+1];
	}
	if(pIni==pEnd)
	{
		if(threadIdx.x==0) GM->splitScore=0.0;
		return;//nothing to do
	}
	__syncthreads();

	double aux;
	long long int pos;
	float x_n[dimsImage];
	float2 rnkAux;

	//double logZp=log(sqrt(determinantSymmetricW_3D(GM->W_k)*pow(GM->nu_k,dimsImage))/pow_2Pi_dimsImage2);


	for(long long int ii=pIni+threadIdx.x;ii<pEnd;ii+=MAX_THREADS_CUDA)//to have colescent access indCUDA
	{
		rnkAux=rnkIndCUDA[ii];
		aux=(double)rnkAux.x;
		if(aux>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
		{
			pos=(long long int)(rnkAux.y);
			x_n[0]=queryCUDA[pos];
			pos+=query_nb;
			x_n[1]=queryCUDA[pos];
			pos+=query_nb;
			x_n[2]=queryCUDA[pos];

			auxJ_k[threadIdx.x]+=aux*(log(aux)+0.5*mahalanobisDistanceCUDA(GM, x_n ));
		}
	}
	__syncthreads();

	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i) auxJ_k[cacheIndex] += auxJ_k[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	aux=N_k[blockIdx.x];
	if (cacheIndex == 0 && aux>0.0)
	{
		double logZp=0.5*log(determinantSymmetricW_3D(GM->W_k)*pow(GM->nu_k,dimsImage)/pow_2Pi_dimsImage);
		GM->splitScore = (auxJ_k[0]/aux)-log(aux)-logZp;
	}

		}

//=======================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMnormalizeXkKernel(float *X_k,float *N_k,int ref_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid<ref_nb && N_k[tid]>0.0f)
	{
		int pos=tid;
		//loop unroll
		X_k[pos]/=N_k[tid];
		pos+=ref_nb;
		X_k[pos]/=N_k[tid];
		pos+=ref_nb;
		X_k[pos]/=N_k[tid];		
	}
				}
//=======================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeSkKernel(int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *X_k,float *S_k,long long int query_nb,int ref_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	long long int pos;
	float x_n[dimsImage],m_k[dimsImage];
	//float imgData;
	float aux;
	int idxAux;

	while(tid<query_nb)
	{
		pos=tid;
		//imgData=imgDataCUDA[pos];
		//loop unrolling for 3D
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			idxAux=indCUDA[pos];//coalescent access
			aux=(float)rnkCUDA[pos];//coalescence access
			if(aux>1e-3)//to avoid unnecessary atomicAdd accesses: rnk has been normalized so it should be high
			{
				//aux*=imgData;
				//loop unroll
				m_k[0]=x_n[0]-X_k[idxAux];
				atomicAddCUDA(&(S_k[idxAux]),aux*m_k[0]*m_k[0]);
				idxAux+=ref_nb;
				m_k[1]=x_n[1]-X_k[idxAux];
				atomicAddCUDA(&(S_k[idxAux]),aux*m_k[0]*m_k[1]);
				idxAux+=ref_nb;
				m_k[2]=x_n[2]-X_k[idxAux];
				atomicAddCUDA(&(S_k[idxAux]),aux*m_k[0]*m_k[2]);
				idxAux+=ref_nb;
				atomicAddCUDA(&(S_k[idxAux]),aux*m_k[1]*m_k[1]);
				idxAux+=ref_nb;
				atomicAddCUDA(&(S_k[idxAux]),aux*m_k[1]*m_k[2]);
				idxAux+=ref_nb;
				atomicAddCUDA(&(S_k[idxAux]),aux*m_k[2]*m_k[2]);				
			}			
			pos+=query_nb;
		}

		//update pointer for next query_point to check
		tid+=offset;
	}
				}

//=======================================================================
//this kernel has to be launch with MAX_THREADS (and with as large of a number as possible)
__global__ void __launch_bounds__(MAX_THREADS_CUDA) GMEMcomputeSkKernelNoAtomic(int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *X_k,float *S_k,long long int query_nb,int ref_nb)
				{

	__shared__ float auxS_k0[MAX_THREADS_CUDA];
	__shared__ float auxS_k1[MAX_THREADS_CUDA];
	__shared__ float auxS_k2[MAX_THREADS_CUDA];
	__shared__ float auxS_k3[MAX_THREADS_CUDA];
	__shared__ float auxS_k4[MAX_THREADS_CUDA];
	__shared__ float auxS_k5[MAX_THREADS_CUDA];


	//reset counter
	auxS_k0[threadIdx.x]=0.0f;
	auxS_k1[threadIdx.x]=0.0f;
	auxS_k2[threadIdx.x]=0.0f;
	auxS_k3[threadIdx.x]=0.0f;
	auxS_k4[threadIdx.x]=0.0f;
	auxS_k5[threadIdx.x]=0.0f;
	__syncthreads();

	float aux;
	long long int pos;
	float x_n[dimsImage],m_k[dimsImage];

	//loop unroll
	pos=blockIdx.x;
	x_n[0]=X_k[pos];
	pos+=ref_nb;
	x_n[1]=X_k[pos];
	pos+=ref_nb;
	x_n[2]=X_k[pos];


	for(long long int ii=threadIdx.x;ii<query_nb*maxGaussiansPerVoxel;ii+=MAX_THREADS_CUDA)//to have colescent access indCUDA
	{
		if(indCUDA[ii]==blockIdx.x)//each block of threads processes a specific Gaussian
		{
			aux=(float)rnkCUDA[ii];
			if(aux>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
			{
				pos=ii%query_nb;
				m_k[0]=x_n[0]-queryCUDA[pos];
				auxS_k0[threadIdx.x]+=aux*m_k[0]*m_k[0];
				pos+=query_nb;
				m_k[1]=x_n[1]-queryCUDA[pos];
				auxS_k1[threadIdx.x]+=aux*m_k[0]*m_k[1];
				pos+=query_nb;
				m_k[2]=x_n[2]-queryCUDA[pos];
				auxS_k2[threadIdx.x]+=aux*m_k[0]*m_k[2];
				pos+=query_nb;
				auxS_k3[threadIdx.x]+=aux*m_k[1]*m_k[1];
				pos+=query_nb;
				auxS_k4[threadIdx.x]+=aux*m_k[1]*m_k[2];
				pos+=query_nb;
				auxS_k5[threadIdx.x]+=aux*m_k[2]*m_k[2];
			}
		}
	}

	__syncthreads();

	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code

	pos=blockIdx.x;
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i) auxS_k0[cacheIndex] += auxS_k0[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) S_k[pos] = auxS_k0[0];//we don't need normalization later on!!
	__syncthreads();
	//----------------------------------
	pos+=ref_nb;
	i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) auxS_k1[cacheIndex] += auxS_k1[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) S_k[pos] = auxS_k1[0];//we don't need normalization later on!!
	__syncthreads();
	//----------------------------------
	pos+=ref_nb;
	i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) auxS_k2[cacheIndex] += auxS_k2[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) S_k[pos] = auxS_k2[0];//we don't need normalization later on!!
	__syncthreads();
	//----------------------------------
	pos+=ref_nb;
	i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) auxS_k3[cacheIndex] += auxS_k3[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) S_k[pos] = auxS_k3[0];//we don't need normalization later on!!
	__syncthreads();
	//----------------------------------
	pos+=ref_nb;
	i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) auxS_k4[cacheIndex] += auxS_k4[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) S_k[pos] = auxS_k4[0];//we don't need normalization later on!!
	__syncthreads();
	//----------------------------------
	pos+=ref_nb;
	i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) auxS_k5[cacheIndex] += auxS_k5[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) S_k[pos] = auxS_k5[0];//we don't need normalization later on!!
	//----------------------------------

				}

//=======================================================================
//We really don't need this kernel since the only time we use S_k is N_k*S_k
__global__ void __launch_bounds__(MAX_THREADS) GMEMnormalizeSkKernel(float *S_k,float *N_k,int ref_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid<ref_nb && N_k[tid]>0.0f)
	{
		int pos=tid;
		//loop unroll
		S_k[pos]/=N_k[tid];
		pos+=ref_nb;
		S_k[pos]/=N_k[tid];
		pos+=ref_nb;
		S_k[pos]/=N_k[tid];
		pos+=ref_nb;
		S_k[pos]/=N_k[tid];
		pos+=ref_nb;
		S_k[pos]/=N_k[tid];
		pos+=ref_nb;
		S_k[pos]/=N_k[tid];
	}
				}
//=======================================================================

//this kernel has to be launch with MAX_THREADS (and with as large of a number as possible)
//calculate mean and covariance in a single pass over teh data
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeXkNkSkKernelTr(int *indCUDAtr,float *queryCUDA,float2 *rnkIndCUDA,int *csrPtrRowACUDA,float *X_k,float *N_k,float *S_k,long long int query_nb,int ref_nb)
				{

	__shared__ float auxX_k0[MAX_THREADS];
	__shared__ float auxX_k1[MAX_THREADS];
	__shared__ float auxX_k2[MAX_THREADS];
	__shared__ float auxN_k[MAX_THREADS];

	__shared__ float auxS_k0[MAX_THREADS];
	__shared__ float auxS_k1[MAX_THREADS];
	__shared__ float auxS_k2[MAX_THREADS];
	__shared__ float auxS_k3[MAX_THREADS];
	__shared__ float auxS_k4[MAX_THREADS];
	__shared__ float auxS_k5[MAX_THREADS];

	__shared__ int pIni;
	__shared__ int pEnd;

	if(threadIdx.x==0)
	{
		pIni=csrPtrRowACUDA[blockIdx.x];
		pEnd=csrPtrRowACUDA[blockIdx.x+1];
	}
	__syncthreads();

	if(pIni==pEnd) return;//nothing to do


	//reset counter
	auxS_k0[threadIdx.x]=0.0f;
	auxS_k1[threadIdx.x]=0.0f;
	auxS_k2[threadIdx.x]=0.0f;
	auxS_k3[threadIdx.x]=0.0f;
	auxS_k4[threadIdx.x]=0.0f;
	auxS_k5[threadIdx.x]=0.0f;

	//reset counter
	auxX_k0[threadIdx.x]=0.0f;
	auxX_k1[threadIdx.x]=0.0f;
	auxX_k2[threadIdx.x]=0.0f;
	auxN_k[threadIdx.x]=0.0f;
	__syncthreads();

	float aux,aux2;
	long long int pos;
	float x_n[dimsImage];
	float2 rnkAux;

	for(long long int ii=pIni+threadIdx.x;ii<pEnd;ii+=MAX_THREADS)//to have colescent access indCUDA
	{
		rnkAux=rnkIndCUDA[ii];
		aux=rnkAux.x;
		if(aux>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
		{
			auxN_k[threadIdx.x]+=aux;
			pos=(long long int)(rnkAux.y);
			x_n[0]=queryCUDA[pos];//non-coalescent access (maybe I should use constant memory)
			pos+=query_nb;
			x_n[1]=queryCUDA[pos];
			pos+=query_nb;
			x_n[2]=queryCUDA[pos];

			aux2=aux*x_n[0];
			auxX_k0[threadIdx.x]+=aux2;
			auxS_k0[threadIdx.x]+=aux2*x_n[0];
			auxS_k1[threadIdx.x]+=aux2*x_n[1];
			auxS_k2[threadIdx.x]+=aux2*x_n[2];

			aux2=aux*x_n[1];
			auxX_k1[threadIdx.x]+=aux2;
			auxS_k3[threadIdx.x]+=aux2*x_n[1];
			auxS_k4[threadIdx.x]+=aux2*x_n[2];

			aux2=aux*x_n[2];
			auxX_k2[threadIdx.x]+=aux2;
			auxS_k5[threadIdx.x]+=aux2*x_n[2];
		}

	}

	__syncthreads();

	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i)
		{ 
			auxN_k[cacheIndex] += auxN_k[cacheIndex + i];
			auxX_k0[cacheIndex] += auxX_k0[cacheIndex + i];
			auxX_k1[cacheIndex] += auxX_k1[cacheIndex + i];
			auxX_k2[cacheIndex] += auxX_k2[cacheIndex + i];
			auxS_k0[cacheIndex] += auxS_k0[cacheIndex + i];
			auxS_k1[cacheIndex] += auxS_k1[cacheIndex + i];
			auxS_k2[cacheIndex] += auxS_k2[cacheIndex + i];
			auxS_k3[cacheIndex] += auxS_k3[cacheIndex + i];
			auxS_k4[cacheIndex] += auxS_k4[cacheIndex + i];
			auxS_k5[cacheIndex] += auxS_k5[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) N_k[blockIdx.x] = auxN_k[0];
	__syncthreads();
	//----------------------------------
	aux=auxN_k[0];
	if(aux>0.0f)
	{
		pos=blockIdx.x+cacheIndex*ref_nb;
		switch(cacheIndex)
		{
		case 0:
			aux2=auxX_k0[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k0[0]-aux2*auxX_k0[0];
			break;
		case 1:
			aux2=auxX_k1[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k1[0]-aux2*auxX_k0[0];
			break;
		case 2:
			aux2=auxX_k2[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k2[0]-aux2*auxX_k0[0];
			break;
		case 3:
			S_k[pos] = auxS_k3[0]-auxX_k1[0]*auxX_k1[0]/aux;
			break;
		case 4:
			S_k[pos] = auxS_k4[0]-auxX_k1[0]*auxX_k2[0]/aux;
			break;
		case 5:
			S_k[pos] = auxS_k5[0]-auxX_k2[0]*auxX_k2[0]/aux;
			break;
		}
	}

				}

//=======================================================================
//this kernel has to be launch with MAX_THREADS (and with as large of a number as possible)
//calculate mean and covariance in a single pass over the data
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeXkNkSkKernelTrWithSupervoxels(float* imgDataCUDA,long long int* labelListPtrCUDA,int *indCUDAtr,float *queryCUDA,float2 *rnkIndCUDA,int *csrPtrRowACUDA,float *X_k,float *N_k,float *S_k,long long int query_nb,int ref_nb)
{
	__shared__ float auxX_k0[MAX_THREADS];
	__shared__ float auxX_k1[MAX_THREADS];
	__shared__ float auxX_k2[MAX_THREADS];
	__shared__ float auxN_k[MAX_THREADS];

	__shared__ float auxS_k0[MAX_THREADS];
	__shared__ float auxS_k1[MAX_THREADS];
	__shared__ float auxS_k2[MAX_THREADS];
	__shared__ float auxS_k3[MAX_THREADS];
	__shared__ float auxS_k4[MAX_THREADS];
	__shared__ float auxS_k5[MAX_THREADS];

	__shared__ int pIni;
	__shared__ int pEnd;
	__shared__ int pIniVoxels;
	__shared__ int pEndVoxels;

	__shared__ float2 rnkAux;//now rnk is common to all voxels within a supervoxels

	if(threadIdx.x==0)
	{
		pIni=csrPtrRowACUDA[blockIdx.x];
		pEnd=csrPtrRowACUDA[blockIdx.x+1];
	}
	__syncthreads();

	if(pIni==pEnd) return;//nothing to do


	//reset counter
	auxS_k0[threadIdx.x]=0.0f;
	auxS_k1[threadIdx.x]=0.0f;
	auxS_k2[threadIdx.x]=0.0f;
	auxS_k3[threadIdx.x]=0.0f;
	auxS_k4[threadIdx.x]=0.0f;
	auxS_k5[threadIdx.x]=0.0f;

	//reset counter
	auxX_k0[threadIdx.x]=0.0f;
	auxX_k1[threadIdx.x]=0.0f;
	auxX_k2[threadIdx.x]=0.0f;
	auxN_k[threadIdx.x]=0.0f;
	__syncthreads();

	float aux,aux2,auxOrig;
	long long int pos;
	float x_n[dimsImage];

	for(long long int ii=pIni;ii<pEnd;ii++)//csrPTRRowACUDA indicates which super-voxels are associated to each Gaussian. It is expected that very few supervoxels are associated to each Gaussian so we prefer to parallelize the inner for loop where we cheack each voxel within a super-voxel
	{
		if(threadIdx.x==0)
		{
			rnkAux=rnkIndCUDA[ii];
		}
		__syncthreads();
		auxOrig=rnkAux.x;//rnk value

		if(auxOrig>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
		{
			if(threadIdx.x==0)//figure out limits for this region
			{
				pIniVoxels=labelListPtrCUDA[(long long int)(rnkAux.y)];
				pEndVoxels=labelListPtrCUDA[((long long int)(rnkAux.y))+1];
			}
			__syncthreads();

			for(long long int jj=pIniVoxels+threadIdx.x;jj<pEndVoxels;jj+=MAX_THREADS)//to have colescent access to imgDataCUDA
			{
				
				//read imgValue and multiply it by rnk since they always appear together
				aux=auxOrig*imgDataCUDA[jj];//even if rnkAux.x is the same all the time, I can not reuse aux because it will be rnkAux.x*ImgDataCUDA[jj]*ImgDataCUDA[jj+MAX_THREADS]*....  The issue is opnly present for regions with area>MAX_THREADS
				//calculate sufficient statistics
				auxN_k[threadIdx.x]+=aux;
				pos=jj;
				x_n[0]=queryCUDA[pos];//non-coalescent access (maybe I should use constant memory)
				pos+=query_nb;
				x_n[1]=queryCUDA[pos];
				pos+=query_nb;
				x_n[2]=queryCUDA[pos];

				aux2=aux*x_n[0];
				auxX_k0[threadIdx.x]+=aux2;
				auxS_k0[threadIdx.x]+=aux2*x_n[0];
				auxS_k1[threadIdx.x]+=aux2*x_n[1];
				auxS_k2[threadIdx.x]+=aux2*x_n[2];

				aux2=aux*x_n[1];
				auxX_k1[threadIdx.x]+=aux2;
				auxS_k3[threadIdx.x]+=aux2*x_n[1];
				auxS_k4[threadIdx.x]+=aux2*x_n[2];

				aux2=aux*x_n[2];
				auxX_k2[threadIdx.x]+=aux2;
				auxS_k5[threadIdx.x]+=aux2*x_n[2];

			}
		}
		__syncthreads();
	}
		
	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i)
		{ 
			auxN_k[cacheIndex] += auxN_k[cacheIndex + i];
			auxX_k0[cacheIndex] += auxX_k0[cacheIndex + i];
			auxX_k1[cacheIndex] += auxX_k1[cacheIndex + i];
			auxX_k2[cacheIndex] += auxX_k2[cacheIndex + i];
			auxS_k0[cacheIndex] += auxS_k0[cacheIndex + i];
			auxS_k1[cacheIndex] += auxS_k1[cacheIndex + i];
			auxS_k2[cacheIndex] += auxS_k2[cacheIndex + i];
			auxS_k3[cacheIndex] += auxS_k3[cacheIndex + i];
			auxS_k4[cacheIndex] += auxS_k4[cacheIndex + i];
			auxS_k5[cacheIndex] += auxS_k5[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) N_k[blockIdx.x] = auxN_k[0];
	__syncthreads();



	//----------------------------------
	aux=auxN_k[0];
	if(aux>0.0f)
	{
		pos=blockIdx.x+cacheIndex*ref_nb;
		switch(cacheIndex)
		{
		case 0:
			aux2=auxX_k0[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k0[0]-aux2*auxX_k0[0];
			break;
		case 1:
			aux2=auxX_k1[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k1[0]-aux2*auxX_k0[0];
			break;
		case 2:
			aux2=auxX_k2[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k2[0]-aux2*auxX_k0[0];
			break;
		case 3:
			S_k[pos] = auxS_k3[0]-auxX_k1[0]*auxX_k1[0]/aux;
			break;
		case 4:
			S_k[pos] = auxS_k4[0]-auxX_k1[0]*auxX_k2[0]/aux;
			break;
		case 5:
			S_k[pos] = auxS_k5[0]-auxX_k2[0]*auxX_k2[0]/aux;
			break;
		}
	}
}
//=======================================================================
/*
//this kernel has to be launch with MAX_THREADS (and with as large of a number as possible)
//calculate mean and covariance in a single pass over teh data
__global__ void __launch_bounds__(MAX_THREADS_CUDA) GMEMcomputeXkNkSkKernelNoAtomic(int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *X_k,float *N_k,float *S_k,long long int query_nb,int ref_nb)
		{

	__shared__ float auxX_k0[MAX_THREADS_CUDA];
	__shared__ float auxX_k1[MAX_THREADS_CUDA];
	__shared__ float auxX_k2[MAX_THREADS_CUDA];
	__shared__ float auxN_k[MAX_THREADS_CUDA];

	__shared__ float auxS_k0[MAX_THREADS_CUDA];
		__shared__ float auxS_k1[MAX_THREADS_CUDA];
		__shared__ float auxS_k2[MAX_THREADS_CUDA];
		__shared__ float auxS_k3[MAX_THREADS_CUDA];
		__shared__ float auxS_k4[MAX_THREADS_CUDA];
		__shared__ float auxS_k5[MAX_THREADS_CUDA];


		//reset counter
		auxS_k0[threadIdx.x]=0.0f;
		auxS_k1[threadIdx.x]=0.0f;
		auxS_k2[threadIdx.x]=0.0f;
		auxS_k3[threadIdx.x]=0.0f;
		auxS_k4[threadIdx.x]=0.0f;
		auxS_k5[threadIdx.x]=0.0f;

	//reset counter
	auxX_k0[threadIdx.x]=0.0f;
	auxX_k1[threadIdx.x]=0.0f;
	auxX_k2[threadIdx.x]=0.0f;
	auxN_k[threadIdx.x]=0.0f;
	__syncthreads();

	float aux,aux2;
	long long int pos;
	float x_n[dimsImage];

	for(long long int ii=threadIdx.x;ii<query_nb*maxGaussiansPerVoxel;ii+=MAX_THREADS_CUDA)//to have colescent access indCUDA
	{
		if(indCUDA[ii]==blockIdx.x)//each block of threads processes a specific Gaussian
		{
			aux=(float)rnkCUDA[ii];
			if(aux>1e-3) //to avoid unnecessary memory reads: rnk has been normalized so it should be high
			{
				auxN_k[threadIdx.x]+=aux;
				pos=ii%query_nb;
				x_n[0]=queryCUDA[pos];
				pos+=query_nb;
				x_n[1]=queryCUDA[pos];
				pos+=query_nb;
				x_n[2]=queryCUDA[pos];

				aux2=aux*x_n[0];
				auxX_k0[threadIdx.x]+=aux2;
				auxS_k0[threadIdx.x]+=aux2*x_n[0];
				auxS_k1[threadIdx.x]+=aux2*x_n[1];
				auxS_k2[threadIdx.x]+=aux2*x_n[2];

				aux2=aux*x_n[1];
				auxX_k1[threadIdx.x]+=aux2;
				auxS_k3[threadIdx.x]+=aux2*x_n[1];
				auxS_k4[threadIdx.x]+=aux2*x_n[2];

				aux2=aux*x_n[2];
				auxX_k2[threadIdx.x]+=aux2;
				auxS_k5[threadIdx.x]+=aux2*x_n[2];
			}
		}
	}

	__syncthreads();

	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex = threadIdx.x;
	while (i != 0) 
	{
		if (cacheIndex < i)
		{ 
			auxN_k[cacheIndex] += auxN_k[cacheIndex + i];
			auxX_k0[cacheIndex] += auxX_k0[cacheIndex + i];
			auxX_k1[cacheIndex] += auxX_k1[cacheIndex + i];
			auxX_k2[cacheIndex] += auxX_k2[cacheIndex + i];
			auxS_k0[cacheIndex] += auxS_k0[cacheIndex + i];
			auxS_k1[cacheIndex] += auxS_k1[cacheIndex + i];
			auxS_k2[cacheIndex] += auxS_k2[cacheIndex + i];
			auxS_k3[cacheIndex] += auxS_k3[cacheIndex + i];
			auxS_k4[cacheIndex] += auxS_k4[cacheIndex + i];
			auxS_k5[cacheIndex] += auxS_k5[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0) N_k[blockIdx.x] = auxN_k[0];
	__syncthreads();
	//----------------------------------
	aux=auxN_k[0];
	if(aux>0.0f)
	{
		pos=blockIdx.x+cacheIndex*ref_nb;
		switch(cacheIndex)
		{
		case 0:
			aux2=auxX_k0[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k0[0]-aux2*auxX_k0[0];
			break;
		case 1:
			aux2=auxX_k1[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k1[0]-aux2*auxX_k0[0];
			break;
		case 2:
			aux2=auxX_k2[0]/aux;
			X_k[pos] = aux2;
			S_k[pos] = auxS_k2[0]-aux2*auxX_k0[0];
			break;
		case 3:
			S_k[pos] = auxS_k3[0]-auxX_k1[0]*auxX_k1[0]/aux;
			break;
		case 4:
			S_k[pos] = auxS_k4[0]-auxX_k1[0]*auxX_k2[0]/aux;
			break;
		case 5:
			S_k[pos] = auxS_k5[0]-auxX_k2[0]*auxX_k2[0]/aux;
			break;
		}
	}

}

 */

//=======================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMupdateGaussianParametersKernel(GaussianMixtureModelCUDA *vecGMCUDA,float *X_k,float *S_k,float *N_k,int ref_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;	

	if(tid<ref_nb)
	{
		GaussianMixtureModelCUDA *GM=&(vecGMCUDA[tid]);

		if(GM->m_o[0]>-1e31 && GM->fixed==false)//cell is alive and it is not fixed
		{
			float auxNk=N_k[tid];
			float auxBeta_o=GM->beta_o,auxBeta_k;
			float x_k[dimsImage];
			float m_o[dimsImage];
			double W_o_inverse[dimsImage*(dimsImage+1)/2];

			//loop unroll
			x_k[0]=X_k[tid];
			tid+=ref_nb;
			x_k[1]=X_k[tid];
			tid+=ref_nb;
			x_k[2]=X_k[tid];

			m_o[0]=GM->m_o[0];
			m_o[1]=GM->m_o[1];
			m_o[2]=GM->m_o[2];

			//update simple scalars
			GM->beta_k=auxBeta_o+auxNk;
			auxBeta_k=GM->beta_k;
			GM->nu_k=GM->nu_o+auxNk;
			GM->alpha_k=fmax(GM->alpha_o+auxNk,0.0);//alpha_o can be negative (improper Dirichlet prior)


			//loop unroll
			GM->m_k[0]=(auxBeta_o*m_o[0]+auxNk*x_k[0])/auxBeta_k;
			GM->m_k[1]=(auxBeta_o*m_o[1]+auxNk*x_k[1])/auxBeta_k;
			GM->m_k[2]=(auxBeta_o*m_o[2]+auxNk*x_k[2])/auxBeta_k;


			//loop unroll
			tid=threadIdx.x + blockIdx.x * blockDim.x;

			if( GM->W_o[5] < 1e-8)//2D case, to avoid 
			{
				GM->W_o[5] = 0.5 * (GM->W_o[0] + GM->W_o[3]);
			}

			inverseSymmetricW_3D(GM->W_o,W_o_inverse);
			//loop unroll for W_k=(W_o.inverse()+N_k*S_k+(auxBeta_o*N_k/(auxBeta_o+N_k))*(x_k-m_o)*(x_k-m_o).transpose()).inverse();
			x_k[0]-=m_o[0];x_k[1]-=m_o[1];x_k[2]-=m_o[2];
			auxBeta_o/=(auxBeta_o+auxNk);//precompute

			//VIP:we assume S_k has not been normalized previously so we save operations and improve precision
			W_o_inverse[0]+=S_k[tid]+auxNk*auxBeta_o*x_k[0]*x_k[0];
			tid+=ref_nb;
			W_o_inverse[1]+=S_k[tid]+auxNk*auxBeta_o*x_k[0]*x_k[1];
			tid+=ref_nb;
			W_o_inverse[2]+=S_k[tid]+auxNk*auxBeta_o*x_k[0]*x_k[2];
			tid+=ref_nb;
			W_o_inverse[3]+=S_k[tid]+auxNk*auxBeta_o*x_k[1]*x_k[1];
			tid+=ref_nb;
			W_o_inverse[4]+=S_k[tid]+auxNk*auxBeta_o*x_k[1]*x_k[2];
			tid+=ref_nb;
			W_o_inverse[5]+=S_k[tid]+auxNk*auxBeta_o*x_k[2]*x_k[2];

			
			inverseSymmetricW_3D(W_o_inverse,GM->W_k);		
		}
	}
}
//-===================================================================
//VIP: THIS FUNCTION HAS TO MATCH GaussianMixtureModel::regularizePrecisionMatrix(void)
//needed for W regularization
//__device__ static const double lambdaMin=0.02;//aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0 and maxRadius=10 (adjust with scale)
//__device__ static const double lambdaMax=0.2222;//aux=scaleSigma/(maxRadius*maxRadius) with scaleSigma=2.0 and minRadius=3.0 (adjust with scale)  (when nuclei divide they can be very narrow)
//__device__ static const double maxExcentricity=3.0*3.0;//maximum excentricity allowed: sigma[i]=1/sqrt(d[i]). Therefore maxExcentricity needs to be squared to used in terms of radius.

__global__ void __launch_bounds__(MAX_THREADS) GMEMregularizeWkKernel(GaussianMixtureModelCUDA *vecGMCUDA,int ref_nb, bool W4DOF, double lambdaMin, double lambdaMax, double maxExcentricity)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<ref_nb)
	{
		GaussianMixtureModelCUDA *GM=&(vecGMCUDA[tid]);

		if(GM->m_o[0]<-1e31 || GM->fixed==true) return;//dead cell or fixed cell in the mixture

		double auxNu=GM->nu_k;		
		double auxMax=lambdaMax/auxNu;//aux=scaleSigma/(minRadius*minRadius*nu_k) with scaleSigma=2.0
		double auxMin=lambdaMin/auxNu;//aux=scaleSigma/(maxRadius*maxRadius*nu_k) with scaleSigma=2.0

		double W_k[dimsImage*(dimsImage+1)/2];
		//copy to local memory in order to perform operations		
		//NOT ALLOWED INSIDE KERNEL cudaMemcpy(W_k,&(GM->W_k[0]),sizeof(double)*dimsImage*(dimsImage+1)/2,cudaMemcpyDeviceToDevice);
		int count;
		for(int count=0;count<dimsImage*(dimsImage+1)/2;count++) W_k[count]=GM->W_k[count];

		//to adjust for scale: this values is empirical
		//If I don;t do this rescaling, I would have to find which eigenvector corresponds to Z dirction to check for min/max Radius
		//basically, maxRadius_z=scaleGMEMCUDA[0]*maxRadius_x/scaleGMEMCUDA[2] 
		count=0;
		for(int ii=0;ii<dimsImage;ii++)
		{
			W_k[count++]/=scaleGMEMCUDA[ii]*scaleGMEMCUDA[ii];
			for(int jj=ii+1;jj<dimsImage;jj++)
				W_k[count++]/=scaleGMEMCUDA[ii]*scaleGMEMCUDA[jj];
		}

	if ( W4DOF == true)
	{
		W_k[2]=0.0f;
		W_k[4]=0.0f;
	}
		double d[dimsImage],v[dimsImage*dimsImage];
		//calculate eigenvalues and eigenvectors
		eig3(W_k,d,v,tid);//NOTE: if dimsImage!=3 it won't work


		/*
		if(threadIdx.x == 91)
		{
			printf("eigenValues=%g %g %g\n",d[0],d[1],d[2]);
			printf("eigenVector1=%g %g %g\n",v[0],v[1],v[2]);
			printf("eigenVector2=%g %g %g\n",v[3],v[4],v[5]);
			printf("eigenVector3=%g %g %g\n",v[6],v[7],v[8]);
			printf("W_k=[%g %g %g;\n",W_k[0],W_k[1],W_k[2]);
			printf("     %g %g %g;\n",W_k[1],W_k[3],W_k[4]);
			printf("     %g %g %g]\n",W_k[2],W_k[4],W_k[5]);
			printf("auxMax=%g;auxMin=%g\n",auxMax,auxMin);
		}
		*/

		//avoid minimum size
		if(d[0]>auxMax) d[0]=auxMax;
		if(d[1]>auxMax) d[1]=auxMax;
		if(d[2]>auxMax) d[2]=auxMax;

		//avoid maximum size
		if(d[0]<auxMin) d[0]=auxMin;
		if(d[1]<auxMin) d[1]=auxMin;
		if(d[2]<auxMin) d[2]=auxMin;

		//avoid too much excentricity
		auxNu=d[0]/d[1];
		if(auxNu>maxExcentricity) d[0]=maxExcentricity*d[1];
		else if(1./auxNu>maxExcentricity) d[1]=maxExcentricity*d[0];

		auxNu=d[0]/d[2];
		if(auxNu>maxExcentricity) d[0]=maxExcentricity*d[2];
		else if(1./auxNu>maxExcentricity) d[2]=maxExcentricity*d[0];

		auxNu=d[1]/d[2];
		if(auxNu>maxExcentricity) d[1]=maxExcentricity*d[2];
		else if(1./auxNu>maxExcentricity) d[2]=maxExcentricity*d[1];

		//reconstruct W_k=V*D*V'
		W_k[0]=d[0]*v[0]*v[0]+d[1]*v[3]*v[3]+d[2]*v[6]*v[6];
		W_k[3]=d[0]*v[1]*v[1]+d[1]*v[4]*v[4]+d[2]*v[7]*v[7];
		W_k[5]=d[0]*v[2]*v[2]+d[1]*v[5]*v[5]+d[2]*v[8]*v[8];

		W_k[1]=d[0]*v[0]*v[1]+d[1]*v[3]*v[4]+d[2]*v[6]*v[7];

		if (W4DOF == false )
		{
			W_k[2]=d[0]*v[0]*v[2]+d[1]*v[3]*v[5]+d[2]*v[6]*v[8];
			W_k[4]=d[0]*v[1]*v[2]+d[1]*v[4]*v[5]+d[2]*v[7]*v[8];
		}

		//undo adjust for scale: 		
		//to adjust for scale: this values is empirical
		//If I don't do this rescaling, I would have to find which eigenvector corresponds to Z dirction to check for min/max Radius
		count=0;
		for(int ii=0;ii<dimsImage;ii++)
		{
			W_k[count++]*=scaleGMEMCUDA[ii]*scaleGMEMCUDA[ii];
			for(int jj=ii+1;jj<dimsImage;jj++)
				W_k[count++]*=scaleGMEMCUDA[ii]*scaleGMEMCUDA[jj];
		}
		//copy results back to global memory
		//cudaMemcpy(&(GM->W_k[0]),W_k,sizeof(double)*dimsImage*(dimsImage+1)/2,cudaMemcpyDeviceToDevice);
		for(int count=0;count<dimsImage*(dimsImage+1)/2;count++) GM->W_k[count]=W_k[count];
	}
}

//=======================================================================
//Check for dead cells
__global__ void __launch_bounds__(MAX_THREADS) GMEMcheckDeadCellsKernel(GaussianMixtureModelCUDA *vecGMCUDA,float *N_k,int ref_nb,double totalAlpha)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid<ref_nb)
	{
		GaussianMixtureModelCUDA *GM=&(vecGMCUDA[tid]);
		//check for nan in W
		bool isnanW=false;
		for(int ii=0;ii<dimsImage*(1+dimsImage+1)/2;ii++)
			if(isnan(GM->W_k[ii]))
				isnanW=true;

		if(isnanW==true || (GM->alpha_k/totalAlpha<minPi_kForDead_CUDA && GM->m_o[0]>-1e31 && GM->fixed==false))
		{
			//kill mixture
			N_k[tid]=0.0f;
			GM->alpha_k=0.0;
			for(int ii=0;ii<dimsImage;ii++)
			{
				GM->m_o[ii]=-1e32;
				GM->m_k[ii]=-1e32;
				GM->W_k[0]=1;GM->W_k[1]=0;GM->W_k[2]=0;GM->W_k[3]=1;GM->W_k[4]=0;GM->W_k[5]=1;//to avoid NaN from regularization
			}
		}
	}
				}
//=======================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeLikelihoodKernel(float *likelihoodVecCUDA,GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,float *imgDataCUDA,long long int query_nb,int ref_nb,double totalAlpha)
		{
	__shared__ float cache[MAX_THREADS];
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//int offset = blockDim.x * gridDim.x;

	long long int pos;
	float x_n[dimsImage];
	double aux,auxLikelihood;
	GaussianMixtureModelCUDA *GM;

	//reset values
	cache[threadIdx.x]=0.0f;
	__syncthreads();

	if(tid>=query_nb) return;

	auxLikelihood=0.0;
	pos=tid;
	//loop unrolling for 3D
	x_n[0]=queryCUDA[pos];
	pos+=query_nb;
	x_n[1]=queryCUDA[pos];
	pos+=query_nb;
	x_n[2]=queryCUDA[pos];

	pos=tid;
	for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
	{
		GM=&(vecGMCUDA[indCUDA[pos]]);
		aux=sqrt(determinantSymmetricW_3D(GM->W_k)*pow(GM->nu_k,dimsImage))/pow_2Pi_dimsImage2;
		aux*=exp(-0.5*mahalanobisDistanceCUDA(GM, x_n ));
		aux*=(GM->alpha_k/totalAlpha);
		//likelihoodPerPixel[n]+=vecPi_k[k]*vecGaussian[k]->eval(x_n);
		auxLikelihood+=aux;

		pos+=query_nb;
	}
	if(auxLikelihood>0.0f)
		cache[threadIdx.x]=imgDataCUDA[tid]*log(auxLikelihood);

	//likelihoodTempCUDA[tid]=(float)auxLikelihood;

	__syncthreads();

	// Max query_nb=65536*MAX_THREADS
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex=threadIdx.x;
	while (i != 0)
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) likelihoodVecCUDA[blockIdx.x] = cache[0];

		}
//=======================================================================
//since all the voxels in the same region are associated with the same Gaussian, we are going to save a lot of memory accesses by computing the likelihood of each super-voxel within each block and then generating a second kernel to add the likelihood from all the super-voxels
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeLikelihoodKernelWithSupervoxels(float *likelihoodVecCUDA,GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,float *imgDataCUDA,long long int *labelListPtrCUDA,long long int query_nb,unsigned short int numLabels,double totalAlpha)
{

	__shared__ float cache[MAX_THREADS];//stores partial sums of log-likelihood
	//If all threads read from the same shared memory address then a broadcast mechanism is automatically invoked and serialization is avoided. Shared memory broadcasts are an excellent and high-performance way to get data to many threads simultaneously.
	__shared__ GaussianMixtureModelCUDA vecGMlabel[maxGaussiansPerVoxel];//stores the data for each Gaussian. We use one block per super-voxel so all threads in the same block access the same Gaussians
	long long int pos=threadIdx.x;
	float x_n[dimsImage];
	double auxLikelihood,aux;

	//reset counter
	cache[pos]=0.0f;
	if(pos<maxGaussiansPerVoxel)
	{
		vecGMlabel[pos]=vecGMCUDA[indCUDA[blockIdx.x+pos*numLabels]];
	}
	__syncthreads();

	for(long long int tid=threadIdx.x+labelListPtrCUDA[blockIdx.x];tid<labelListPtrCUDA[blockIdx.x+1];tid+=MAX_THREADS)
	{
		pos=tid;
		//loop unrolling for 3D
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		auxLikelihood=0;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{			
			aux=sqrt(determinantSymmetricW_3D(vecGMlabel[ii].W_k)*pow(vecGMlabel[ii].nu_k,dimsImage))/pow_2Pi_dimsImage2;
			aux*=exp(-0.5*mahalanobisDistanceCUDA(&(vecGMlabel[ii]), x_n ));
			aux*=(vecGMlabel[ii].alpha_k/totalAlpha);
			//likelihoodPerPixel[n]+=vecPi_k[k]*vecGaussian[k]->eval(x_n);
			auxLikelihood+=aux;			
		}
		
		if(auxLikelihood>0.0f)
			cache[threadIdx.x]=imgDataCUDA[tid]*log(auxLikelihood);				
	}

	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex=threadIdx.x;
	while (i != 0)
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) likelihoodVecCUDA[blockIdx.x] = cache[0];	
}

//=======================================================================
//since all the voxels in the same region are associated with the same Gaussian, we are going to save a lot of memory accesses by computing the likelihood of each super-voxel within each block and then generating a second kernel to add the likelihood from all the super-voxels
//we compute likelihood just over a set of specified supervoxels (local)
//call teh function with numGrids = listSupervoxelsIdxLength
//it is teh same code as GMEMcomputeLikelihoodKernelWithSupervoxels but substituting blockIdx.x by supervoxelIdx
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeLocalLikelihoodKernelWithSupervoxels(float *likelihoodVecCUDA,GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,float *imgDataCUDA,long long int *labelListPtrCUDA,long long int query_nb,int ref_nb, unsigned short int numLabels,double totalAlpha, int* listSupervoxelsIdx, int listSupervoxelsIdxLength)
{

	__shared__ float cache[MAX_THREADS];//stores partial sums of log-likelihood
	//If all threads read from the same shared memory address then a broadcast mechanism is automatically invoked and serialization is avoided. Shared memory broadcasts are an excellent and high-performance way to get data to many threads simultaneously.
	__shared__ GaussianMixtureModelCUDA vecGMlabel[maxGaussiansPerVoxel];//stores the data for each Gaussian. We use one block per super-voxel so all threads in the same block access the same Gaussians
	long long int pos=threadIdx.x;
	float x_n[dimsImage];
	double auxLikelihood,aux;

	if( blockIdx.x >= listSupervoxelsIdxLength )
		return;

	int supervoxelIdx = listSupervoxelsIdx[blockIdx.x];

	//reset counter
	cache[pos]=0.0f;
	if(pos<maxGaussiansPerVoxel)
	{
		vecGMlabel[pos]=vecGMCUDA[indCUDA[supervoxelIdx+pos*numLabels]];

		//----------------------------debug: hola2--------------------------
		/*
		if( supervoxelIdx == 7)
		{
			printf("Gaussian:%3d; %12f %12f %12f; %12f %12f %12f %12f %12f %12f;%12f\n",indCUDA[supervoxelIdx+pos*numLabels], vecGMlabel[pos].m_k[0],vecGMlabel[pos].m_k[1], vecGMlabel[pos].m_k[2], vecGMlabel[pos].W_k[0] , vecGMlabel[pos].W_k[1] , vecGMlabel[pos].W_k[2] , vecGMlabel[pos].W_k[3] , vecGMlabel[pos].W_k[4] , vecGMlabel[pos].W_k[5], vecGMlabel[pos].nu_k );
			printf("Parameter for:%3d %1.16f %1.16f %1.16f\n", indCUDA[supervoxelIdx+pos*numLabels],determinantSymmetricW_3D(vecGMlabel[pos].W_k), vecGMlabel[pos].alpha_k/totalAlpha, determinantSymmetricW_3D(vecGMlabel[pos].W_k)*pow(vecGMlabel[pos].nu_k,dimsImage) );
		}
		*/
		//------------------------------------------------
	}
	__syncthreads();

	for(long long int tid=threadIdx.x+labelListPtrCUDA[supervoxelIdx];tid<labelListPtrCUDA[supervoxelIdx+1];tid+=MAX_THREADS)
	{
		pos=tid;
		//loop unrolling for 3D
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		auxLikelihood = 0;
		
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{			
			
			aux=sqrt( determinantSymmetricW_3D(vecGMlabel[ii].W_k)*pow(vecGMlabel[ii].nu_k,dimsImage) )/pow_2Pi_dimsImage2;
			aux*=exp(-0.5*mahalanobisDistanceCUDA(&(vecGMlabel[ii]), x_n ));
			aux*=(vecGMlabel[ii].alpha_k/totalAlpha);
			//likelihoodPerPixel[n]+=vecPi_k[k]*vecGaussian[k]->eval(x_n);
			auxLikelihood+=aux;			
		}
		
		/*looking at all the Gaussians instead of just the nearest neighbors (it should be a little bit more accurate in some cases)
		for(int ii = 0; ii < ref_nb; ii++) //hola2
		{
			aux=sqrt( determinantSymmetricW_3D(vecGMCUDA[ii].W_k)*pow(vecGMCUDA[ii].nu_k,dimsImage) )/pow_2Pi_dimsImage2;
			aux*=exp(-0.5*mahalanobisDistanceCUDA(&(vecGMCUDA[ii]), x_n ));
			aux*=(vecGMCUDA[ii].alpha_k/totalAlpha);
			//likelihoodPerPixel[n]+=vecPi_k[k]*vecGaussian[k]->eval(x_n);
			auxLikelihood+=aux;	
		}
		*/

		if(auxLikelihood>0.0f)
			cache[threadIdx.x]=imgDataCUDA[tid]*log(auxLikelihood);	

		//---debug:hola2-------------------------------------------
		/*
		if(supervoxelIdx == 7)
		{
			if(auxLikelihood>0.0f)
				printf("%4f %4f %4f %10f %10f %1.16f %10f\n",x_n[0],x_n[1],x_n[2],imgDataCUDA[tid],log(auxLikelihood),auxLikelihood,imgDataCUDA[tid]*log(auxLikelihood));			
		}
		*/
	}

	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	int cacheIndex=threadIdx.x;
	while (i != 0)
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) likelihoodVecCUDA[blockIdx.x] = cache[0];	
}




//=====================================================================
//kernel to calculate total lieklihood: based on dot product exmaple
//if(likelihoodPerPixel[ii]>0) logLikelihood+=imgData[ii]*log(likelihoodPerPixel[ii]);
__global__ void __launch_bounds__(MAX_THREADS) GMEMsumLikelihoodKernel(float *imgDataCUDA,float *likelihoodTempCUDA,float *likelihoodVecCUDA,int query_nb)
{
	__shared__ float cache[MAX_THREADS];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float aux;

	// set the cache values
	if(tid<query_nb)
	{
		aux=likelihoodTempCUDA[tid];
		if(aux>0.0f)
			cache[cacheIndex] = imgDataCUDA[tid]*log(aux);
		else cache[cacheIndex] = 0.0;
	}
	else cache[cacheIndex] = 0.0;
	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		likelihoodVecCUDA[blockIdx.x] = cache[0];
}

//=====================================================================
//we sum the elements of a vector (it follows the classical example of dot product on how to perform reduction operations)
//the vector must have length=2^n
//the partial sums are saved in vector[0...numBlocks], so you have to finish some computations on the GPU side
//VIP: the values of the vector are modified with partial sums!!!
__global__ void __launch_bounds__(MAX_THREADS) sumVector(float *vector,long long int length)
{
	__shared__ float cache[MAX_THREADS];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	// set the cache values
	if(tid<length)
	{
		cache[cacheIndex]=vector[tid];
	}else{
		cache[cacheIndex]=0.0f;
	}
	
	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	int i = blockDim.x/2;
	while (i != 0) 
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0)
		vector[blockIdx.x] = cache[0];
}



//=======================================================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMcomputeRnkKernel(GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,pxi *rnkCUDA,float *imgDataCUDA,long long int query_nb)
				{
	// map from threadIdx/BlockIdx to pixel position
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//int offset = blockDim.x * gridDim.x;

	long long int pos;
	float x_n[dimsImage];
	float rnkAux[maxGaussiansPerVoxel];
	float rnkNorm;
	GaussianMixtureModelCUDA *GMCUDA;
	double aux;

	
	while(tid<query_nb)
	{
		pos=tid;
		rnkNorm=0.0f;
		//loop unrolling for 3D
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			GMCUDA=&(vecGMCUDA[indCUDA[pos]]);
			aux=-0.5*expectedMahalanobisDistanceCUDA(GMCUDA,x_n);
			aux+=GMCUDA->expectedLogResponsivityCUDA;
			aux+=0.5*GMCUDA->expectedLogDetCovarianceCUDA;
			if(aux>-100)//exp(-100)=3.7201e-44. We consider zero otherwise
			{
				rnkAux[ii]=(pxi)(exp(aux));
				rnkNorm+=rnkAux[ii];
			}
			else rnkAux[ii]=0.0;
			pos+=query_nb;
		}
		//normalize rnk
		if(rnkNorm>0.0f)
		{
			pos=tid;
			float imgData=imgDataCUDA[tid];
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
			{
				rnkCUDA[pos]=imgData*rnkAux[ii]/rnkNorm;//rnk is always multiplied by imgData so we do it only once and avoid it later
				pos+=query_nb;
			}
		}

		//update pointer for next query_point to check
		//tid+=offset;
		tid+=blockDim.x * gridDim.x;//to reduce number of registers
	}
}
//============================================================================================================================
//the approach for this kernel is very different than the approach for the same kernel without supervoxels since we have to perform reduction techniques in the GPU
//actually the approach is very similar to the kernel GMEMcomputeXkNkKernelNoAtomic where each block deals with a single Gaussian mixture (in this case each blocks deals with a single region)

//VIP: we assume we points are ordered by region assignment, so each thread can access it in a cache friendly manner
__global__ void __launch_bounds__(MAX_THREADS_CUDA) GMEMcomputeRnkKernelWithSupervoxels(GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,pxi *rnkCUDA,long long int *labelListPtrCUDA,long long int query_nb,unsigned short int numLabels)
{

	__shared__ float rnkVec[MAX_THREADS_CUDA*maxGaussiansPerVoxel];//it stores partial accumulative values of rnk for a regions
	//If all threads read from the same shared memory address then a broadcast mechanism is automatically invoked and serialization is avoided. Shared memory broadcasts are an excellent and high-performance way to get data to many threads simultaneously.
	__shared__ GaussianMixtureModelCUDA vecGMlabel[maxGaussiansPerVoxel];//stores the data for each Gaussian. We use one block per super-voxel so all threads in the same block access the same Gaussians
	//reset counter
	int ii=0;
	int offset=threadIdx.x;
	for(ii=0;ii<maxGaussiansPerVoxel;ii++)
	{
		rnkVec[offset]=0.0f;
		offset+=MAX_THREADS_CUDA;
	}
	if(threadIdx.x<maxGaussiansPerVoxel)
	{
		vecGMlabel[threadIdx.x]=vecGMCUDA[indCUDA[blockIdx.x+threadIdx.x*numLabels]];
	}
	__syncthreads();


	long long int pos;
	float x_n[dimsImage];
	double aux;


	for(long long int tid=threadIdx.x+labelListPtrCUDA[blockIdx.x];tid<labelListPtrCUDA[blockIdx.x+1];tid+=MAX_THREADS_CUDA)
	{
		pos=tid;
		//loop unrolling for 3D
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		offset=threadIdx.x;
		for(ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			aux=-0.5*expectedMahalanobisDistanceCUDA(&(vecGMlabel[ii]),x_n);
			aux+=vecGMlabel[ii].expectedLogResponsivityCUDA;
			aux+=0.5*vecGMlabel[ii].expectedLogDetCovarianceCUDA;
			rnkVec[offset]+=aux;//adding logarithm quantities
			offset+=MAX_THREADS_CUDA;//offset=threadIdx.x+ii*MAX_THREADS_CUDA	
		}		

	}

	__syncthreads();



	//reduction by adding all the quantities at rnkVec
	
	//final addition for each __shared__ memory vector
	// for reductions, threadsPerBlock must be a power of 2
	// because of the following code
	
	offset=0;
	for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
	{
		ii = blockDim.x/2 ;
		
		pos = threadIdx.x+offset;//offset=jj*MAX_THREADS_CUDA
		while (ii != 0) 
		{
			if (pos < ii+offset) 
				rnkVec[pos] += rnkVec[pos + ii];
			__syncthreads();
			ii /= 2;
		}
		offset+=MAX_THREADS_CUDA;
		__syncthreads();
	}

	//normalize rnkVec and save in the output variable
	//check notebook July 10th 2012 + wikipedia for logarithm addition http://en.wikipedia.org/wiki/List_of_logarithmic_identities without losing precision
	if (threadIdx.x <maxGaussiansPerVoxel) 
	{
		aux=0.0;
		double aux2=rnkVec[threadIdx.x*MAX_THREADS_CUDA];
		for(ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			aux+=exp(rnkVec[ii*MAX_THREADS_CUDA]-aux2);			
		}
		aux=1.0/aux;
		rnkCUDA[blockIdx.x+threadIdx.x*numLabels] = aux;	
	}	
}
//============================================================================================================================
__global__ void GMEMprintGaussianMixtureModelKernel(GaussianMixtureModelCUDA *vecGMCUDA,float *refTempCUDA,int idx)
{

#ifdef USE_CUDA_PRINTF
	GaussianMixtureModelCUDA *GM=&(vecGMCUDA[idx]);

	cuPrintf("beta_k=%g;alpha_k=%g;nu_k=%g\n",GM->beta_k,GM->alpha_k,GM->nu_k);
	cuPrintf("m_k=%g %g %g\n",GM->m_k[0],GM->m_k[1],GM->m_k[2]);
	cuPrintf("m_k(constant mem buffer)=%g %g %g\n",refTempCUDA[3*idx],refTempCUDA[3*idx+1],refTempCUDA[3*idx+2]);
	cuPrintf("W_k=%g %g %g %g %g %g\n",GM->W_k[0],GM->W_k[1],GM->W_k[2],GM->W_k[3],GM->W_k[4],GM->W_k[5]);

	cuPrintf("beta_o=%g;alpha_o=%g;nu_o=%g\n",GM->beta_o,GM->alpha_o,GM->nu_o);
	cuPrintf("m_o=%g %g %g\n",GM->m_o[0],GM->m_o[1],GM->m_o[2]);
	cuPrintf("W_o=%g %g %g %g %g %g\n",GM->W_o[0],GM->W_o[1],GM->W_o[2],GM->W_o[3],GM->W_o[4],GM->W_o[5]);

	cuPrintf("expectedLogDet=%g;expectedLogResp=%g\n",GM->expectedLogDetCovarianceCUDA,GM->expectedLogResponsivityCUDA);


#endif
}
void printGaussianMixtureModelCUDA(GaussianMixtureModelCUDA *vecGMCUDA,float *refTempCUDA,int idx)
{
#ifdef USE_CUDA_PRINTF
	cudaPrintfInit();

	GMEMprintGaussianMixtureModelKernel<<<1,1>>>(vecGMCUDA,refTempCUDA,idx);
	HANDLE_ERROR_KERNEL;
	printf("=============================\n");
	cudaPrintfDisplay(stdout, true);
	printf("=============================\n");
	cudaPrintfEnd();
#endif
}
//==========================================================================================================================
//slight modification from Nvidia SDK transpose matrix since our matrices are very skinny
//dim3 grid(min(65536,(int)(query_nb+TILE_DIM-1)/TILE_DIM), 1), threads(TILE_DIM,1);
__global__  void __launch_bounds__(TILE_DIM) GMEMtransposeArrayInt(int *idata,int *odata,long long int query_nb)
		{
	__shared__ int tile[TILE_DIM][maxGaussiansPerVoxel];

	//read idata from global memory to share memory
	int row=blockIdx.x*blockDim.x;//we need to execute whole blocks at a time	
	int offset=blockDim.x*gridDim.x;
	const int aux=TILE_DIM/maxGaussiansPerVoxel;
	const long long int aux2=query_nb*maxGaussiansPerVoxel;

	while(row<query_nb)//we need to execute whole blocks at a time
	{   
		int pos=row+threadIdx.x;

		if(pos<query_nb)
		{
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
			{
				tile[threadIdx.x][ii] = idata[pos];
				pos+=query_nb;
			}
		}else{
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++) tile[threadIdx.x][ii]=0.0f;
		}
		__syncthreads();

		int n=threadIdx.x%maxGaussiansPerVoxel;
		int p=threadIdx.x/maxGaussiansPerVoxel;
		pos=maxGaussiansPerVoxel*blockDim.x*blockIdx.x+threadIdx.x;
		if(pos<(query_nb-TILE_DIM)*maxGaussiansPerVoxel)
		{
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
			{
				odata[pos] = tile[p][n];//TILE_DIM is a multiple of maxGaussians per voxel
				pos+=TILE_DIM;
				p+=aux;
			}
		}else{//write one by one 
			while(pos<aux2)
			{
				odata[pos] = tile[p][n];//TILE_DIM is a multiple of maxGaussians per voxel
				pos+=TILE_DIM;
				p+=aux;	    		
			}
		}
		__syncthreads();
		row+=offset;
	}


		}
//==========================================================================================================================
//slight modification from Nvidia SDK transpose matrix since our matrices are very skinny
//dim3 grid(min(65536,(int)(query_nb+TILE_DIM-1)/TILE_DIM), 1), threads(TILE_DIM,1);
__global__  void __launch_bounds__(TILE_DIM) GMEMtransposeArrayFloat(float *idata,float *odata,long long int query_nb)
		{
	__shared__ float tile[TILE_DIM][maxGaussiansPerVoxel];

	//read idata from global memory to share memory
	int row=blockIdx.x*blockDim.x;//we need to execute whole blocks at a time	
	int offset=blockDim.x*gridDim.x;
	const int aux=TILE_DIM/maxGaussiansPerVoxel;
	const long long int aux2=query_nb*maxGaussiansPerVoxel;

	while(row<query_nb)//we need to execute whole blocks at a time
	{   
		int pos=row+threadIdx.x;

		if(pos<query_nb)
		{
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
			{
				tile[threadIdx.x][ii] = idata[pos];
				pos+=query_nb;
			}
		}else{
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++) tile[threadIdx.x][ii]=0.0f;
		}
		__syncthreads();

		int n=threadIdx.x%maxGaussiansPerVoxel;
		int p=threadIdx.x/maxGaussiansPerVoxel;
		pos=maxGaussiansPerVoxel*blockDim.x*blockIdx.x+threadIdx.x;
		if(pos<(query_nb-TILE_DIM)*maxGaussiansPerVoxel)
		{
			for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
			{
				odata[pos] = tile[p][n];//TILE_DIM is a multiple of maxGaussians per voxel
				pos+=TILE_DIM;
				p+=aux;
			}
		}else{//write one by one 
			while(pos<aux2)
			{
				odata[pos] = tile[p][n];//TILE_DIM is a multiple of maxGaussians per voxel
				pos+=TILE_DIM;
				p+=aux;	    		
			}
		}
		__syncthreads();
		row+=offset;
	}


		}

//=========================================
__global__  void GMEMsetcscColPtrA(int *cscColPtrACUDA,long long int arrayLength)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid<arrayLength)
	{
		cscColPtrACUDA[tid]=maxGaussiansPerVoxel*tid;
	}
}
__global__  void GMEMsetcsrColIndA(int *csrColIndACUDA,float2 *rnkIndCUDA,long long int query_nb)
{
	int auxInd[maxGaussiansPerVoxel];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	float2 aux;

	while(tid<query_nb)
	{

		//read in
		long long int pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			aux=rnkIndCUDA[pos];
			auxInd[ii]=(int)(aux.y);
			pos+=query_nb;
		}
		//write out
		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			csrColIndACUDA[pos]=auxInd[ii];
			pos+=query_nb;
		}

		tid+=offset;
	}
}
__global__  void GMEMsetRnktrKernel(float *rnkCUDAtr,float2 *rnkIndCUDA,long long int query_nb)
{
	float auxInd[maxGaussiansPerVoxel];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while(tid<query_nb)
	{

		//read in
		long long int pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			auxInd[ii]=rnkIndCUDA[pos].x;
			pos+=query_nb;
		}
		//write out
		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			rnkCUDAtr[pos]=auxInd[ii];
			pos+=query_nb;
		}

		tid+=offset;
	}
}

__global__  void GMEMsetcsrRowPtrAKernel(int *csrRowPtrACUDA,int *indCUDAtr,long long int query_nb,int ref_nb)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	int x0[maxGaussiansPerVoxel],x1[maxGaussiansPerVoxel];

	while(tid<query_nb-1)
	{

		//read in
		long long int pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			x0[ii]=indCUDAtr[pos];
			x1[ii]=indCUDAtr[pos+1];
			pos+=query_nb;
		}
		//write out
		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			if(x0[ii]!=x1[ii])
			{
				csrRowPtrACUDA[x0[ii]+1]=pos+1;
			}
			pos+=query_nb;
		}

		tid+=offset;
	}
	if(tid==query_nb-1)//special case
	{
		long long int pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel-1;ii++)
		{
			x0[ii]=indCUDAtr[pos];
			x1[ii]=indCUDAtr[pos+1];
			pos+=query_nb;
		}
		//save last index at the end of the vector to handle final elements
		csrRowPtrACUDA[ref_nb]=indCUDAtr[pos];
		//write out
		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel-1;ii++)
		{
			if(x0[ii]!=x1[ii])
			{
				csrRowPtrACUDA[x0[ii]+1]=pos+1;
			}
			pos+=query_nb;
		}
	}
}

void GMEMsetcsrRowPtrA(int *csrRowPtrAHost,int *csrRowPtrACUDA,int *indCUDAtr,long long int query_nb,int ref_nb)
{
	long long int numThreads_query=std::min((long long int)MAX_THREADS,query_nb);
	long long int numGrids_query=std::min((long long int)MAX_BLOCKS,(query_nb+numThreads_query-1)/numThreads_query);
	const long long int auxCt=maxGaussiansPerVoxel*query_nb;
	//printf("NumGrids=%d;numThreads=%d\n",numThreads_query,numGrids_query);

	//reset all to zero
	HANDLE_ERROR(cudaMemset(csrRowPtrACUDA,0,sizeof(int)*(ref_nb+1)));
	//set pointer where there is an index difference
	GMEMsetcsrRowPtrAKernel<<<numGrids_query,numThreads_query>>>(csrRowPtrACUDA,indCUDAtr,query_nb,ref_nb);HANDLE_ERROR_KERNEL;
	//finish in host to fill in the gaps
	HANDLE_ERROR(cudaMemcpy(csrRowPtrAHost,csrRowPtrACUDA,sizeof(int)*(ref_nb+1),cudaMemcpyDeviceToHost));

	int lastVal=csrRowPtrAHost[ref_nb];//needed to handle end of the csrRowPtr
	//printf("====DEBUGGING: at void GMEMsetcsrRowPtrA. Incorrect lastVal = %d and ref_nb = %d\n",lastVal, ref_nb);
	if(lastVal>=ref_nb)
	{
		printf("ERROR: at void GMEMsetcsrRowPtrA. Incorrect lastVal = %d and ref_nb = %d\n",lastVal, ref_nb);
		exit(3);
	}
	for(int ii=lastVal+1;ii<=ref_nb;ii++) csrRowPtrAHost[ii]=auxCt;
	//---------------------debug------------------------
	/*
	printf("DEBUGGING: debugging col compress pointer");
	FILE *pFile=fopen("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1311858269_CPU/rowPtrBefore.txt","w");
		for(int ii=0;ii<ref_nb+1;ii++)
		{			
			fprintf(pFile," %d",csrRowPtrAHost[ii]);					
		}		
		fprintf(pFile,";\n");
	 */	
	//------------------------------------------------

	int lastElem=0,aux;
	for(int ii=1;ii<ref_nb;ii++)
	{
		aux=csrRowPtrAHost[ii];
		if(aux>0)
		{
			lastElem=aux;
		}else csrRowPtrAHost[ii]=lastElem;
	}

	//copy back to device
	HANDLE_ERROR(cudaMemcpy(csrRowPtrACUDA,csrRowPtrAHost,sizeof(int)*(ref_nb+1),cudaMemcpyHostToDevice));

	//--------------------------debug----------------------------------------------------
	/*
	printf("DEBUGGING: debugging col compress pointer");
	int *indHost=new int[maxGaussiansPerVoxel*query_nb];
	HANDLE_ERROR( cudaMemcpy( indHost, indCUDAtr, sizeof(int) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );


	pFile=fopen("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1311858269_CPU/indTr.txt","w");
	for(int ii=0;ii<query_nb*maxGaussiansPerVoxel;ii++)
	{			
		fprintf(pFile," %d",indHost[ii]);					
	}		
	fprintf(pFile,";\n");
	pFile=fopen("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1311858269_CPU/rowPtr.txt","w");
	for(int ii=0;ii<ref_nb+1;ii++)
	{			
		fprintf(pFile," %d",csrRowPtrAHost[ii]);					
	}		
	fprintf(pFile,";\n");
	exit(2);
	 */
	//-----------------------------------------------------------------------------------

}
//======================================================================================================
__global__ void GMEMcopyRnkBeforeSortKernel(float *rnkCUDA,float2 *rnkIndCUDA,long long int query_nb)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	float2 aux[maxGaussiansPerVoxel];
	while(tid<query_nb)
	{
		long long int pos=tid;
		float id=(float)tid;
		//write in
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			aux[ii].x=rnkCUDA[pos];
			aux[ii].y=id;
			pos+=query_nb;
		}
		//write out
		pos=tid;
		for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
		{
			rnkIndCUDA[pos]=aux[ii];
			pos+=query_nb;
		}
		tid+=offset;
	}
}

//=========================================
__global__ void __launch_bounds__(MAX_THREADS) GMEMsetMatrixBkernel(float *B,float *queryCUDA,long long int query_nb)
		{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;
	float x_n[dimsImage];
	while(tid<query_nb)
	{
		long long int pos=tid;
		x_n[0]=queryCUDA[pos];
		pos+=query_nb;
		x_n[1]=queryCUDA[pos];
		pos+=query_nb;
		x_n[2]=queryCUDA[pos];

		//B is row-major order (great for coalescent write)
		pos=tid;
		B[pos]=1.0f;
		pos+=query_nb;
		B[pos]=x_n[0];
		pos+=query_nb;
		B[pos]=x_n[1];
		pos+=query_nb;
		B[pos]=x_n[2];
		pos+=query_nb;
		B[pos]=x_n[0]*x_n[0];
		pos+=query_nb;
		B[pos]=x_n[0]*x_n[1];
		pos+=query_nb;
		B[pos]=x_n[0]*x_n[2];
		pos+=query_nb;
		B[pos]=x_n[1]*x_n[1];
		pos+=query_nb;
		B[pos]=x_n[1]*x_n[2];
		pos+=query_nb;
		B[pos]=x_n[2]*x_n[2];

		tid+=offset;
	}
		}

//===========================================================================================================================
void allocateMemoryForGaussianMixtureModelCUDA(GaussianMixtureModelCUDA **vecGMCUDA,pxi **rnkCUDA,double **totalAlphaTempCUDA,long long int query_nb,int ref_nb,int numGridsCheck)
{
	HANDLE_ERROR( cudaMalloc( (void**)&(*vecGMCUDA), ref_nb*sizeof(GaussianMixtureModelCUDA) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*rnkCUDA), query_nb*maxGaussiansPerVoxel*sizeof(pxi) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*totalAlphaTempCUDA), numGridsCheck*sizeof(double) ) );
}

void updateMemoryForGaussianMixtureModelCUDA(GaussianMixtureModelCUDA *vecGMCUDAtemp,GaussianMixtureModelCUDA *vecGMCUDA,int ref_nb)
{
	HANDLE_ERROR( cudaMemcpy( vecGMCUDA, vecGMCUDAtemp, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyHostToDevice ) );
}

void deallocateMemoryForGaussianMixtureModelCUDA(GaussianMixtureModelCUDA **vecGMCUDA,pxi **rnkCUDA,double **totalAlphaTempCUDA)
{
	HANDLE_ERROR( cudaFree( *vecGMCUDA ) );
	(*vecGMCUDA)=NULL;
	HANDLE_ERROR( cudaFree( *rnkCUDA ) );
	(*rnkCUDA)=NULL;
	HANDLE_ERROR( cudaFree( *totalAlphaTempCUDA ) );
	(*totalAlphaTempCUDA)=NULL;
}
//=============================================================
double calculateTotalAlpha(GaussianMixtureModelCUDA *vecGMCUDA,double *totalAlphaTempCUDA,double *totalAlphaTemp,int ref_nb,int numGridsCheck)
{
	//calculate number of threads and blocks
	int numThreads=std::min(MAX_THREADS,ref_nb);
	int numGrids=std::min(MAX_BLOCKS,(ref_nb+numThreads-1)/numThreads);

	if(numGrids==1)//make sure numThreads is a power of 2
	{
		numThreads=(int)pow(2.0f,(int)ceil(log2((float)numThreads)));
	}

	if(numGrids!=numGridsCheck)
	{
		printf("ERROR: numGrids does not agree!\n");
		exit(2);
	}
	//printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);
	dim3    grids(numGrids,1);
	dim3    threads(numThreads,1);
	GMEMtotalAlphaKernel<<<grids,threads>>>(vecGMCUDA,totalAlphaTempCUDA,ref_nb);
	HANDLE_ERROR_KERNEL;

	//finish adding sum
	HANDLE_ERROR(cudaMemcpy(totalAlphaTemp,totalAlphaTempCUDA,numGrids*sizeof(double),cudaMemcpyDeviceToHost));//retrieve partial sum

	double totalAlpha=totalAlphaTemp[0];
	for(int ii=1;ii<numGrids;ii++) totalAlpha+=totalAlphaTemp[ii];

	//printf("totalAlpha=%g\n",totalAlpha);

	return totalAlpha;
}
//=============================================================
double addTotalLikelihood(float *imgDataCUDA,float *likelihoodVecCUDA,float* finalSumVectorInHostF,GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,long long int query_nb,int ref_nb,double totalAlpha)
{
	//calculate number of threads and blocks
	long long int numThreads=std::min((long long )MAX_THREADS,query_nb);
	long long int numGrids=std::min((long long )MAX_BLOCKS,(query_nb+numThreads-1)/numThreads);

	if(numGrids==1)//make sure numThreads is a power of 2
	{
		numThreads=(int)pow(2.0f,(int)ceil(log2((float)numThreads)));
	}

	GMEMcomputeLikelihoodKernel<<<numGrids,numThreads>>>(likelihoodVecCUDA,vecGMCUDA,indCUDA,queryCUDA,imgDataCUDA,query_nb,ref_nb,totalAlpha);HANDLE_ERROR_KERNEL;


	//GMEMsumLikelihoodKernel<<<numGrids,numThreads>>>(imgDataCUDA,likelihoodTempCUDA,likelihoodVecCUDA,query_nb);
	HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaMemcpy( finalSumVectorInHostF, likelihoodVecCUDA, sizeof(float) * numGrids, cudaMemcpyDeviceToHost ) );
	double ll=finalSumVectorInHostF[0];
	for(int ii=1;ii<numGrids;ii++) 
	{	
		//printf("%g ",finalSumVectorInHostF[ii]);
		ll+=finalSumVectorInHostF[ii];
	}

	return ll;
}
//=============================================================
double addTotalLikelihoodWithSupervoxels(float *imgDataCUDA,float *likelihoodVecCUDA,float* finalSumVectorInHostF,GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,long long int *labelListPtrCUDA,long long int query_nb,unsigned short int numLabels,double totalAlpha)
{
	//calculate number of threads and blocks
	int numThreads=MAX_THREADS;
	int numGrids=numLabels;

	int sizeLikelihoodVec=(int)pow(2.0f,(int)ceil(log2((float)numLabels)));

	HANDLE_ERROR(cudaMemset(likelihoodVecCUDA,0,sizeof(float)*sizeLikelihoodVec));//some elements might not be updated since we chose the smallest power of 2 above numLabels
	
	GMEMcomputeLikelihoodKernelWithSupervoxels<<<numGrids,numThreads>>>(likelihoodVecCUDA,vecGMCUDA,indCUDA,queryCUDA,imgDataCUDA,labelListPtrCUDA,query_nb,numLabels,totalAlpha);HANDLE_ERROR_KERNEL;
	
		
	//add likelihood from different regions
	numThreads=std::min(MAX_THREADS,sizeLikelihoodVec);
	numGrids=(sizeLikelihoodVec+numThreads-1)/numThreads;
	sumVector<<<numGrids,numThreads>>>(likelihoodVecCUDA,sizeLikelihoodVec);HANDLE_ERROR_KERNEL;

	
	
	
	//finish the addition in the CPU
	HANDLE_ERROR( cudaMemcpy( finalSumVectorInHostF, likelihoodVecCUDA, sizeof(float) * numGrids, cudaMemcpyDeviceToHost ) );
	double ll=finalSumVectorInHostF[0];
	for(int ii=1;ii<numGrids;ii++) 
	{	
		//printf("%g ",finalSumVectorInHostF[ii]);
		ll+=finalSumVectorInHostF[ii];
	}

	return ll;
}

//=============================================================
double addLocalLikelihoodWithSupervoxels(float *imgDataCUDA,float *likelihoodVecCUDA,float* finalSumVectorInHostF,GaussianMixtureModelCUDA *vecGMCUDA,int *indCUDA,float *queryCUDA,long long int *labelListPtrCUDA,long long int query_nb,int ref_nb,unsigned short int numLabels,double totalAlpha, int* listSupervoxelsIdxCUDA, int listSupervoxelsIdxLength)
{
	//calculate number of threads and blocks
	int numThreads = MAX_THREADS;
	int numGrids = listSupervoxelsIdxLength; //one block per supervoxel

	int numLocalLabels = listSupervoxelsIdxLength;


	if(numLocalLabels == 0)//to protect from error launching the kernel. It is just an empty Gaussian
		return -1e32;//very unlikely event

	int sizeLikelihoodVec=(int)pow(2.0f,(int)ceil(log2((float)numLocalLabels)));

	HANDLE_ERROR(cudaMemset(likelihoodVecCUDA,0,sizeof(float)*sizeLikelihoodVec));//some elements might not be updated since we chose the smallest power of 2 above numLabels

	GMEMcomputeLocalLikelihoodKernelWithSupervoxels<<<numGrids,numThreads>>>(likelihoodVecCUDA,vecGMCUDA,indCUDA,queryCUDA,imgDataCUDA,labelListPtrCUDA,query_nb,ref_nb,numLabels,totalAlpha, listSupervoxelsIdxCUDA, listSupervoxelsIdxLength);
	
	char errMsgInfo[512];
	sprintf(errMsgInfo,"numGrids=%d; numThreads=%d\n",numGrids,numThreads);
	HANDLE_ERROR_KERNEL_MSG(errMsgInfo);
	

	//-------------------debug----------------------------
	/*
		cout<<"DEBUGGING: printing out local likelihood for each surpevoxel. listSupervoxelsIdxLength="<<listSupervoxelsIdxLength<<endl;
		int *listSupervoxelsIdx = new int[sizeLikelihoodVec];//upper bound
		memset(listSupervoxelsIdx, 0, sizeLikelihoodVec * sizeof(int) );
		float *likelihoodVec = new float[ sizeLikelihoodVec ];
		HANDLE_ERROR( cudaMemcpy( listSupervoxelsIdx, listSupervoxelsIdxCUDA, sizeof(int) * numGrids, cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( likelihoodVec, likelihoodVecCUDA, sizeof(float) * sizeLikelihoodVec, cudaMemcpyDeviceToHost ) );

		for(int ii =0 ; ii<sizeLikelihoodVec; ii++)
		{
			printf("%d %g \n",listSupervoxelsIdx[ii], likelihoodVec[ii]);
		}
		delete[] listSupervoxelsIdx;
		delete[] likelihoodVec;

		exit(3);
	*/	
	//----------------------------------------------------
		

			
	//add likelihood from different regions
	numThreads=std::min(MAX_THREADS,sizeLikelihoodVec);
	numGrids=(sizeLikelihoodVec+numThreads-1)/numThreads;
	sumVector<<<numGrids,numThreads>>>(likelihoodVecCUDA,sizeLikelihoodVec);HANDLE_ERROR_KERNEL;
	
	
	
	//finish the addition in the CPU
	HANDLE_ERROR( cudaMemcpy( finalSumVectorInHostF, likelihoodVecCUDA, sizeof(float) * numGrids, cudaMemcpyDeviceToHost ) );
	double ll=finalSumVectorInHostF[0];
	for(int ii=1;ii<numGrids;ii++) 
	{					
		ll+=finalSumVectorInHostF[ii];
	}


	return ll;
}

//==============================================================================================================
void calculateLocalLikelihood(vector<double>& localLikelihood, const vector< vector<int> >& listSupervoxels, float *queryCUDA,float *imgDataCUDA,pxi *rnkCUDA,int *indCUDA,GaussianMixtureModelCUDA *vecGMtemp,long long int *labelListPtrCUDA,long long int query_nb,int ref_nb, int numLabels)
{
	localLikelihood.resize( listSupervoxels.size() );

	//find maximum number of local labels needed to precallocate memory
	int numLocalLabelsMax = 0;
	for(size_t ii = 0; ii < listSupervoxels.size(); ii++)
		numLocalLabelsMax = max(numLocalLabelsMax, (int) (listSupervoxels[ii].size()) );

	//calculate numThreads and numGrids for different kernels
	int numThreads_ref=std::min(MAX_THREADS,ref_nb);
	int numGrids_ref=std::min(MAX_BLOCKS,(ref_nb+numThreads_ref-1)/numThreads_ref);
	long long 	int numThreads_query=std::min((long long int)MAX_THREADS,query_nb);
	long long 	int numGrids_query=std::min((long long int)MAX_BLOCKS,(query_nb+numThreads_query-1)/numThreads_query);
	long long 	int numThreads_labels=std::min((long long int)MAX_THREADS,(long long int)numLocalLabelsMax);
	long long 	int numGrids_labels=std::min((long long int)MAX_BLOCKS,(long long int)((numLocalLabelsMax+numThreads_labels-1)/numThreads_labels));

	int sizeLikelihoodVec=(int)pow(2.0f,(int)ceil(log2((float)numLocalLabelsMax)));//we need it to be apower of 2 in order to perform summation as a kernel (dot product example)
	long long 	int numThreads_likelihood=std::min((long long int)MAX_THREADS,(long long int)sizeLikelihoodVec);
	long long 	int numGrids_likelihood=std::min((long long int)MAX_BLOCKS,(long long int)((sizeLikelihoodVec+numThreads_likelihood-1)/numThreads_likelihood));


	//allocate memory in host
	double* finalSumVectorInHostD=(double*)malloc(sizeof(double)*numGrids_ref);
	float* finalSumVectorInHostF=(float*)malloc(sizeof(float)*numGrids_likelihood);


	//allocate memory in device
	GaussianMixtureModelCUDA *vecGMCUDA;
	double* totalAlphaTempCUDA;
	float* likelihoodVecCUDA;
	int* listSupervoxelsIdxCUDA;

	HANDLE_ERROR( cudaMalloc( (void**)&(vecGMCUDA), ref_nb*sizeof(GaussianMixtureModelCUDA) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(totalAlphaTempCUDA), numGrids_ref*sizeof(double) ) );	
	HANDLE_ERROR( cudaMalloc( (void**)&likelihoodVecCUDA, sizeLikelihoodVec*sizeof(float) ) );//we allocate a maximum amount of memory need (usually it is smaller since we use a small subset of supervoxels to calculate likelihood
	HANDLE_ERROR( cudaMalloc( (void**)&listSupervoxelsIdxCUDA, numLocalLabelsMax * sizeof(int) ) );//we allocate a maximum amount of memory needed (usually, it is smaller)

	//upload vecGM info
	HANDLE_ERROR( cudaMemcpy( vecGMCUDA, vecGMtemp, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyHostToDevice ) );

	
	//calculate total alpha
	double totalAlpha = calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);

	//main for loop: calculate local likelihood for each element
	for(size_t ii = 0; ii < localLikelihood.size(); ii++)
	{
		if( listSupervoxels[ii].empty() == true )
			continue;
		HANDLE_ERROR( cudaMemcpy( listSupervoxelsIdxCUDA, &(listSupervoxels[ii][0]), sizeof(int) * listSupervoxels[ii].size(), cudaMemcpyHostToDevice ) );
		localLikelihood[ii] = addLocalLikelihoodWithSupervoxels(imgDataCUDA,likelihoodVecCUDA,finalSumVectorInHostF,vecGMCUDA,indCUDA,queryCUDA,labelListPtrCUDA,query_nb, ref_nb,numLabels, totalAlpha, listSupervoxelsIdxCUDA, listSupervoxels[ii].size() );		
	}

	//release memory
	HANDLE_ERROR( cudaFree(likelihoodVecCUDA ) );
	HANDLE_ERROR(cudaFree(totalAlphaTempCUDA));
	HANDLE_ERROR(cudaFree(vecGMCUDA));

	
	//release host memory
	free(finalSumVectorInHostD);
	free(finalSumVectorInHostF);
	
}

//===================================================================================================
void GMEMcomputeRnkCUDA(pxi *rnk,pxi *rnkCUDA,int *indCUDA,float *queryCUDA,long long int query_nb,float* imgDataCUDA,GaussianMixtureModelCUDA *vecGMCUDA)
{

	//calculate number of threads and blocks
	long long int numThreads=std::min((long long int)MAX_THREADS,query_nb);
	long long int numGrids=std::min((long long int)MAX_BLOCKS,(query_nb+numThreads-1)/numThreads);
	//printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);
	dim3    grids(numGrids,1);
	dim3    threads(numThreads,1);
#ifdef USE_CUDA_PRINTF
	cudaPrintfInit();
#endif

	HANDLE_ERROR(cudaMemset(rnkCUDA,0,sizeof(float)*maxGaussiansPerVoxel*query_nb));//some rnk might not be set
	GMEMcomputeRnkKernel<<<grids,threads>>>(vecGMCUDA,indCUDA,queryCUDA,rnkCUDA,imgDataCUDA,query_nb);
	HANDLE_ERROR_KERNEL;

#ifdef USE_CUDA_PRINTF
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
#endif

	HANDLE_ERROR(cudaMemcpy(rnk,rnkCUDA,query_nb*maxGaussiansPerVoxel*sizeof(pxi),cudaMemcpyDeviceToHost));//retrieve r_nk
}
//===================================================================================================
void GMEMcomputeRnkCUDAInplace(pxi *rnkCUDA,int *indCUDA,float *queryCUDA,long long int query_nb,float* imgDataCUDA,GaussianMixtureModelCUDA *vecGMCUDA)
{

	//calculate number of threads and blocks
long long 	int numThreads=std::min((long long int)MAX_THREADS,query_nb);
long long 	int numGrids=std::min((long long int)MAX_BLOCKS,(query_nb+numThreads-1)/numThreads);
	//printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);
	dim3    grids(numGrids,1);
	dim3    threads(numThreads,1);
#ifdef USE_CUDA_PRINTF
	cudaPrintfInit();
#endif

	HANDLE_ERROR(cudaMemset(rnkCUDA,0,sizeof(pxi)*maxGaussiansPerVoxel*query_nb));//some rnk might not be set
	GMEMcomputeRnkKernel<<<grids,threads>>>(vecGMCUDA,indCUDA,queryCUDA,rnkCUDA,imgDataCUDA,query_nb);
	HANDLE_ERROR_KERNEL;

#ifdef USE_CUDA_PRINTF
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
#endif

	//HANDLE_ERROR(cudaMemcpy(rnk,rnkCUDA,query_nb*maxGaussiansPerVoxel*sizeof(pxi),cudaMemcpyDeviceToHost));//retrieve r_nk
}

//===================================================================================================
void GMEMcomputeRnkCUDAInplaceWithSupervoxels(pxi *rnkCUDA,int *indCUDA,float *queryCUDA,long long int query_nb,unsigned short int numLabels,GaussianMixtureModelCUDA *vecGMCUDA,long long int *labelListPtrCUDA)
{
	//calculate number of threads and blocks
	int numThreads=MAX_THREADS_CUDA;
	int numGrids=numLabels;
	//printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);
	dim3    grids(numGrids,1);
	dim3    threads(numThreads,1);

	HANDLE_ERROR(cudaMemset(rnkCUDA,0,sizeof(pxi)*maxGaussiansPerVoxel*numLabels));//some rnk might not be set
	GMEMcomputeRnkKernelWithSupervoxels<<<grids,threads>>>(vecGMCUDA,indCUDA,queryCUDA,rnkCUDA,labelListPtrCUDA,query_nb,numLabels);
	HANDLE_ERROR_KERNEL;


	//HANDLE_ERROR(cudaMemcpy(rnk,rnkCUDA,query_nb*maxGaussiansPerVoxel*sizeof(pxi),cudaMemcpyDeviceToHost));//retrieve r_nk
}

//====================================================================================================
void GMEMupdatePriorConstantsCUDA(GaussianMixtureModelCUDA *vecGMCUDA,int ref_nb,double totalAlphaDiGamma)
{
	//calculate number of threads and blocks
	int numThreads=std::min(MAX_THREADS,ref_nb);
	int numGrids=std::min(MAX_BLOCKS,(ref_nb+numThreads-1)/numThreads);
	//printf("NumThreads=%d;numGrids=%d\n",numThreads,numGrids);
	dim3    grids(numGrids,1);
	dim3    threads(numThreads,1);
#ifdef USE_CUDA_PRINTF
	cudaPrintfInit();
#endif

	GMEMexpectedLogDetCovarianceKernel<<<grids,threads>>>(vecGMCUDA,ref_nb);
	HANDLE_ERROR_KERNEL;
	GMEMexpectedLogResponsivityKernel<<<grids,threads>>>(vecGMCUDA,ref_nb,totalAlphaDiGamma);
	HANDLE_ERROR_KERNEL;
#ifdef USE_CUDA_PRINTF
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
#endif
}

//==============================================================
// initialize cusparse library
void initializeCuSparseLinrary(cusparseHandle_t &handle,cusparseMatDescr_t &descra)
{
	cusparseStatus_t status= cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf("CUSPARSE Library initialization failed\n");
		exit(2);
	}
	//create and setup matrix descriptor
	status= cusparseCreateMatDescr(&descra);
	if (status != CUSPARSE_STATUS_SUCCESS)
	{
		printf("Matrix descriptor initialization failed\n");
		exit(2);
	}
	cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);//not symmetric or anything like this
	cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);
}

//===================================================================================================
/*
int main(int argc, const char* argv[])
{
	if(argc<2) return 1;
	//mainTestGMEMcomputeRnkCUDA(atoi(argv[1]));
	mainTestGMEMcomputeVariationalInference(atoi(argv[1]));
	return 0;
}
 */

//==================================================================================================
void mainTestGMEMcomputeRnkCUDA(long long int query_nb)
{
	int iterations=20;
	//long long int query_nb=50000;
	int ref_nb=500;

	int numGridsCheck=(ref_nb+MAX_THREADS-1)/MAX_THREADS;

	int dev=0;
	HANDLE_ERROR( cudaSetDevice( dev ) );

	//host variables
	pxi *rnk;
	int *ind;
	float *query;
	GaussianMixtureModelCUDA *vecGM;
	double *totalAlphaTemp;

	rnk=(pxi*)malloc(maxGaussiansPerVoxel*query_nb*sizeof(pxi));
	query  = (float *) malloc(query_nb * dimsImage * sizeof(float));
	ind    = (int *)   malloc(query_nb * maxGaussiansPerVoxel * sizeof(int));
	vecGM = (GaussianMixtureModelCUDA*)malloc(ref_nb*sizeof(GaussianMixtureModelCUDA));
	totalAlphaTemp = (double*) malloc(numGridsCheck*sizeof(double));

	//generate random points
	// Init 
	srand((unsigned int)time(NULL));
	for (int i=0 ; i<query_nb * dimsImage ; i++) query[i]  = (float)rand() / (float)RAND_MAX;
	for (int i=0 ; i<query_nb * maxGaussiansPerVoxel ; i++) ind[i]  = (int)floor(0.99*ref_nb*(float)rand() / (float)RAND_MAX);

	for(int i=0;i<ref_nb;i++)
	{
		vecGM[i].beta_k=0.1;
		vecGM[i].nu_k=dimsImage+1.0;
		vecGM[i].alpha_k=(((float)rand() / (float)RAND_MAX)-0.5)*100.0;
		for(int j=0;j<dimsImage;j++) vecGM[i].m_k[j]=(float)rand() / (float)RAND_MAX;
		vecGM[i].expectedLogDetCovarianceCUDA=-(float)rand() / (float)RAND_MAX;
		vecGM[i].expectedLogResponsivityCUDA=-(float)rand() / (float)RAND_MAX;

		memset(vecGM[i].W_k,0,dimsImage*(dimsImage+1)*sizeof(double)/2);
		vecGM[i].W_k[0]=0.33;//diagonal
		vecGM[i].W_k[3]=0.1;
		vecGM[i].W_k[5]=0.2;
		vecGM[i].W_k[1]=0.01;
		vecGM[i].W_k[2]=0.02;
		vecGM[i].W_k[4]=0.04;
	}
	memset(rnk,0,maxGaussiansPerVoxel*query_nb*sizeof(pxi));


	//------------------------------------------------------------
	/*
	//print out 
	printf("query=[");
	int idxAux=0;
	for (int i=0 ; i<dimsImage ; i++)
	{
		for(int j=0;j<query_nb;j++)
		{
			printf("%g ",query[idxAux]);
			idxAux++;
		}
		printf(";\n");
	}
	printf("];\n");
	printf("ind=[");
	idxAux=0;
	for (int i=0 ; i<maxGaussiansPerVoxel ; i++)
	{
		for(int j=0;j<query_nb;j++)
		{
			printf("%d ",ind[idxAux]);
			idxAux++;
		}
		printf(";\n");
	}
	printf("];\n");
	printf("logDet=[");
	for (int i=0 ; i<ref_nb ; i++)
	{
		printf("%g ",vecGM[i].expectedLogDetCovarianceCUDA);		
	}
	printf("];\n");
	printf("logResp=[");
	for (int i=0 ; i<ref_nb ; i++)
	{
		printf("%g ",vecGM[i].expectedLogResponsivityCUDA);		
	}
	printf("];\n");
	printf("m_k=[");
	for (int i=0 ; i<ref_nb ; i++)
	{
		for(int j=0;j<dimsImage;j++)
		{
			printf("%g ",vecGM[i].m_k[j]);
		}
		printf(";\n");
	}
	printf("];\n");
	 */
	//----------------------------------------------

	//CUDA variables 
	pxi *rnkCUDA=NULL;
	pxi *rnkCUDAtr=NULL;
	int *indCUDA=NULL;
	float *queryCUDA=NULL;
	GaussianMixtureModelCUDA* vecGMCUDA=NULL;
	double *totalAlphaTempCUDA=NULL;
	float *imgDataCUDA=NULL;

	CUcontext cuContext;
	CUdevice  cuDevice=dev;
	cuCtxCreate(&cuContext, 0, cuDevice);
	size_t memoryNeededInBytes=query_nb*maxGaussiansPerVoxel*sizeof(int)+query_nb*dimsImage*sizeof(float)+maxGaussiansPerVoxel*query_nb*sizeof(pxi)+ref_nb*sizeof(GaussianMixtureModelCUDA);
	size_t memTotal,memAvailable;
	cuMemGetInfo(&memAvailable,&memTotal);
	printf( "Memory required: %lu;CUDA choosing device:  %d;memory available=%lu;total mem=%lu\n", memoryNeededInBytes,dev,memAvailable,memTotal);
	cuCtxDetach (cuContext);

	/*for some reason this reads wrong in TEsla
	if(memAvailable<memoryNeededInBytes)
	{
		printf("ERROR: not enough memory in GPU!!\n");
		exit(2);
	}
	*/
	//allocate memory in device
	allocateMemoryForGaussianMixtureModelCUDA(&vecGMCUDA,&rnkCUDA,&totalAlphaTempCUDA,query_nb,ref_nb,numGridsCheck);
	HANDLE_ERROR( cudaMalloc( (void**)&indCUDA, query_nb*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&queryCUDA, query_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(rnkCUDAtr), query_nb*maxGaussiansPerVoxel*sizeof(pxi) ) );


	HANDLE_ERROR( cudaMemcpy( indCUDA, ind, sizeof(int) * query_nb*maxGaussiansPerVoxel, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( queryCUDA, query, sizeof(float) * query_nb*dimsImage, cudaMemcpyHostToDevice ) );


	// capture the start time
	cudaEvent_t     start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );


	//main calculation
	double totalAlpha;
	for(int ii=0;ii<iterations;ii++)
	{
		updateMemoryForGaussianMixtureModelCUDA(vecGM,vecGMCUDA,ref_nb);
		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,totalAlphaTemp,ref_nb,numGridsCheck);
		GMEMupdatePriorConstantsCUDA(vecGMCUDA,ref_nb,DiGamma(totalAlpha));
		GMEMcomputeRnkCUDA(rnk,rnkCUDA,indCUDA,queryCUDA,query_nb,imgDataCUDA,vecGMCUDA);

	}


	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
	printf(" done in %f secs for %d iterations (%f secs per iteration)\n", elapsedTime/1000, iterations, elapsedTime/(iterations*1000));

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	//print out results
	/*
	printf("Rnk=[");
	for(int ii=0;ii<query_nb;ii++)
	{
		for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
		{
			printf("%g ",rnk[ii+query_nb*jj]);
		}
		printf(";\n");
	}
	printf("];\n");
	 */
	//check result
	float auxNorm;
	for(int ii=0;ii<query_nb;ii++)
	{
		auxNorm=0.0;
		for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
		{
			auxNorm+=rnk[ii+query_nb*jj];
		}
		if(fabs(auxNorm-1.0)>1e-2)
		{
			printf("ERROR: wromng results!\n");
			exit(2);
		}
	}
	//release device memory
	deallocateMemoryForGaussianMixtureModelCUDA(&vecGMCUDA,&rnkCUDA,&totalAlphaTempCUDA);
	HANDLE_ERROR(cudaFree(rnkCUDAtr));

	//release host memory
	free(rnk);
	free(ind);
	free(query);
	free(vecGM);
	free(totalAlphaTemp);
}

//=================================================================================
//main interface to run variational inference in GPU
void GMEMvariationalInferenceCUDA(float *queryCUDA,float *imgDataCUDA,pxi *rnkCUDA,pxi *rnkCUDAtr,int *indCUDA,int *indCUDAtr,GaussianMixtureModelCUDA *vecGMtemp,long long int query_nb,int ref_nb,int maxIterEM,double tolLikelihood,int devCUDA,int frame,bool W4DOF, string debugPath)
{
	if(query_nb>((long long int)MAX_THREADS) *((long long int)MAX_BLOCKS))
	{
		printf("ERROR: Too many query points for the GPU resources\n");//knnCUDA does not have while(tid<ref_nb) to handle infinite query_nb because of syncthreads
		exit(0);
	}
	if(ref_nb>MAX_BLOCKS)
	{
		printf("ERROR: Too many reference points for the GPU resources\n");
		exit(0);
	}

	//host variables
	double *finalSumVectorInHostD;
	float *finalSumVectorInHostF;
	int *csrRowPtrAHost;

	//CUDA variables 
	//pxi *rnkCUDA=NULL;
	//int *indCUDA=NULL;
	GaussianMixtureModelCUDA* vecGMCUDA=NULL;
	double *totalAlphaTempCUDA=NULL;
	float *refTempCUDA=NULL;
	float *X_k=NULL;//maximization step
	float *N_k=NULL;//maximization step
	float *S_k=NULL;//maximization step
	//float *likelihoodTempCUDA=NULL;
	float *likelihoodVecCUDA=NULL;
	int *csrRowPtrACUDA;
	float2 *rnkIndCUDA=NULL;


	//cusparse matrix
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descrA=0;
	initializeCuSparseLinrary(handle,descrA);

	//calculate numThreads and numGrids for different kernels
	int numThreads_ref=std::min(MAX_THREADS,ref_nb);
	int numGrids_ref=std::min(MAX_BLOCKS,(ref_nb+numThreads_ref-1)/numThreads_ref);
long long 	int numThreads_query=std::min((long long int)MAX_THREADS,query_nb);
long long 	int numGrids_query=std::min((long long int)MAX_BLOCKS,(query_nb+numThreads_query-1)/numThreads_query);

	//select GPU and check maximum meory available and check that we have enough memory
	size_t memoryNeededInBytes=ref_nb*sizeof(GaussianMixtureModelCUDA)+
			numGrids_ref*sizeof(double)+
			ref_nb*dimsImage*sizeof(float)+
			ref_nb*(1+dimsImage+dimsImage*(1+dimsImage)/2)*sizeof(float)+
			numGrids_query*sizeof(float)+
			sizeof(int)*(ref_nb+1)+
			sizeof(float2)*query_nb*maxGaussiansPerVoxel;//this is the main load aside from preallocated rnkCUDA,rnkCUDAtr and indCUDA,indCUDAtr

	CUcontext cuContext;
	CUdevice  cuDevice=devCUDA;
	cuCtxCreate(&cuContext, 0, cuDevice);
	size_t memTotal,memAvailable;
	cuMemGetInfo(&memAvailable,&memTotal);
#if defined(_WIN32) || defined(_WIN64)
	printf( "Memory required: %lu;CUDA choosing device:  %Iu;memory available=%Iu;total mem=%Iu\n", memoryNeededInBytes,devCUDA,memAvailable,memTotal);
#else
	printf( "Memory required: %lu;CUDA choosing device:  %zu;memory available=%zu;total mem=%zu\n", memoryNeededInBytes,devCUDA,memAvailable,memTotal);
#endif
	cuCtxDetach (cuContext);

	/*for some reason this reads wrong in TEsla
	if(memAvailable<memoryNeededInBytes)
	{
		printf("ERROR: not enough memory in GPU!!\n");
		exit(2);
	}
	*/
	// capture the start time
	cudaEvent_t     start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	//allocate memory in host
	finalSumVectorInHostD=(double*)malloc(sizeof(double)*numGrids_ref);
	finalSumVectorInHostF=(float*)malloc(sizeof(float)*numGrids_query);
	csrRowPtrAHost=(int *)malloc(sizeof(int)*(ref_nb+1));

	//allocate memory in device
	HANDLE_ERROR( cudaMalloc( (void**)&(vecGMCUDA), ref_nb*sizeof(GaussianMixtureModelCUDA) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(totalAlphaTempCUDA), numGrids_ref*sizeof(double) ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&indCUDA, query_nb*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&refTempCUDA, ref_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&N_k, ref_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&X_k, ref_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&S_k, ref_nb*dimsImage*(dimsImage+1)*sizeof(float)/2 ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&likelihoodTempCUDA, query_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&likelihoodVecCUDA, numGrids_query*sizeof(float) ) );
	HANDLE_ERROR(cudaMalloc((void**)&csrRowPtrACUDA,sizeof(int)*(ref_nb+1)));
	HANDLE_ERROR(cudaMalloc((void**)&rnkIndCUDA,sizeof(float2)*query_nb*maxGaussiansPerVoxel));

	//wrap pointers for thurst library
	thrust::device_ptr<int> thrust_indCUDAtr(indCUDAtr);//keys
	thrust::device_ptr<float2> thrust_rnkIndCUDA(rnkIndCUDA);//values

	//upload vecGM info
	HANDLE_ERROR( cudaMemcpy( vecGMCUDA, vecGMtemp, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyHostToDevice ) );

	//iterate variational inference for one image
	int numIterEM=1;
	double totalAlpha;
	double ll=-1.0,llOld=ll*10,llOld2=ll*100,llOld3=ll*1000;//sometimes it oscillates between multiple values
	while(fabs(ll-llOld)/fabs(llOld)>tolLikelihood && fabs(ll-llOld2)/fabs(llOld2)>tolLikelihood && fabs(ll-llOld3)/fabs(llOld3)>tolLikelihood && numIterEM<=maxIterEM)
	{
		
		//------------------------debug each iteration and save it in a file (look at the function to change the destination folder)
		//HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
		//writeXMLdebugCUDA(vecGMtemp,ref_nb);
		//---------------------------------------------------------------------------------------------------------
		
		//calculate nearest neighbors
		GMEMcopyGaussianCenter2ConstantMemoryKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,refTempCUDA,ref_nb);HANDLE_ERROR_KERNEL;
		knnCUDAinPlace(indCUDA,queryCUDA,refTempCUDA,query_nb,ref_nb);

		//calculate responsibilities (expectation step)
		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);
		GMEMupdatePriorConstantsCUDA(vecGMCUDA,ref_nb,DiGamma(totalAlpha));
		GMEMcomputeRnkCUDAInplace(rnkCUDA,indCUDA,queryCUDA,query_nb,imgDataCUDA,vecGMCUDA);

		//------------------------debug rnk-----------------------
		/*
		printf("DEBUGGING: rnk on CUDA\n");
		float *rnkHost=new float[maxGaussiansPerVoxel*query_nb];
		int *indHost=new int[maxGaussiansPerVoxel*query_nb];
		float *imgDataHost=new float[query_nb];
		HANDLE_ERROR( cudaMemcpy( rnkHost, rnkCUDA, sizeof(float) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( indHost, indCUDA, sizeof(int) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( imgDataHost, imgDataCUDA, sizeof(float) * query_nb, cudaMemcpyDeviceToHost ) );


		FILE *pFile=fopen("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1311858269_CPU/rnk.txt","w");
		for(int ii=0;ii<query_nb;ii++)
		{
			for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
			{
				fprintf(pFile," %g",rnkHost[ii+query_nb*jj]);
			}
			fprintf(pFile,";\n");
		}		
		fclose(pFile);
		pFile=fopen("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1311858269_CPU/ind.txt","w");
		for(int ii=0;ii<query_nb;ii++)
		{
			for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
			{
				fprintf(pFile," %d",indHost[ii+query_nb*jj]);
			}
			fprintf(pFile,";\n");
		}		
		fclose(pFile);
		 */
		//--------------------------------------------------------

		//-----------------------using no transpose: memory access is awful--------------------------------------
		//calculate new Gaussian parameters (maximization step)
		//HANDLE_ERROR(cudaMemset(N_k,0,ref_nb*sizeof(float)));
		//HANDLE_ERROR(cudaMemset(X_k,0,ref_nb*dimsImage*sizeof(float)));
		//HANDLE_ERROR(cudaMemset(S_k,0,ref_nb*dimsImage*(dimsImage+1)*sizeof(float)/2));
		//GMEMcomputeXkNkKernel<<<numGrids_query,numThreads_query>>>(indCUDA,queryCUDA,rnkCUDA,X_k,N_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;
		//GMEMnormalizeXkKernel<<<numGrids_ref,numThreads_ref>>>(X_k,N_k,ref_nb);HANDLE_ERROR_KERNEL;
		//GMEMcomputeSkKernel<<<numGrids_query,numThreads_query>>>(indCUDA,queryCUDA,rnkCUDA,X_k,S_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;

		//GMEMcomputeXkNkKernelNoAtomic<<<ref_nb,MAX_THREADS_CUDA>>>(indCUDA,queryCUDA,rnkCUDA,X_k,N_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;//it does not require X_k normalization		
		//GMEMcomputeSkKernelNoAtomic<<<ref_nb,MAX_THREADS_CUDA>>>(indCUDA,queryCUDA,rnkCUDA,X_k,S_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;

		//---------------------------------------------------------------------------------------------------

		//---------------------------------------------------------------------------------
		//use thrust library to transpose Rnk matrix
		// wrap raw pointer with a device_ptr
		GMEMcopyRnkBeforeSortKernel<<<numGrids_query,numThreads_query>>>(rnkCUDA,rnkIndCUDA,query_nb);HANDLE_ERROR_KERNEL;

		//amatf
		//HANDLE_ERROR(cudaMemcpy(indCUDAtr,indCUDA,sizeof(int)*maxGaussiansPerVoxel*query_nb,cudaMemcpyDeviceToDevice));//because we need indCUDA later
		memCpyDeviceToDeviceKernel<<<numGrids_query,numThreads_query>>>(indCUDA,indCUDAtr,maxGaussiansPerVoxel*query_nb);HANDLE_ERROR_KERNEL;


		thrust::sort_by_key(thrust_indCUDAtr, thrust_indCUDAtr+maxGaussiansPerVoxel*query_nb, thrust_rnkIndCUDA);//super fast radix sorting (166e6 elements/sec in my GPU) 

		//set row compress index format
		GMEMsetcsrRowPtrA(csrRowPtrAHost,csrRowPtrACUDA,indCUDAtr,query_nb,ref_nb);




		//-------------------------------------using cuSparse: it is slower than our own kernel and needs more memory------------------------------------------------
		/*
		//cusparse level 3 (matrix-matrix multipication)
		 int *csrColIndACUDA=NULL;
		float *B=NULL;
		float *C=NULL;
		HANDLE_ERROR(cudaMalloc((void**)&csrColIndACUDA,sizeof(int)*query_nb*maxGaussiansPerVoxel));
		HANDLE_ERROR(cudaMalloc((void**)&B,sizeof(float)*query_nb*10));
		HANDLE_ERROR(cudaMalloc((void**)&C,sizeof(float)*ref_nb*10));
		HANDLE_ERROR(cudaMemset(C,0,sizeof(float)*ref_nb*10));
		GMEMsetMatrixBkernel<<<numGrids_query,numThreads_query>>>(B,queryCUDA,query_nb);HANDLE_ERROR_KERNEL;//TODO this only needs to be done once per EM iterations, so we could move it outside teh foor loop
		GMEMsetcsrColIndA<<<numGrids_query,numThreads_query>>>(csrColIndACUDA,rnkIndCUDA,query_nb);HANDLE_ERROR_KERNEL;

		if(rnkCUDAtr==NULL)
		{
			printf("ERROR: rnkCUDAtr should not be NULL!\n");
			exit(2);
		}
		//copy rnkIndCUDA indexes into a single vector
		GMEMsetRnktrKernel<<<numGrids_query,numThreads_query>>>(rnkCUDAtr,rnkIndCUDA,query_nb);HANDLE_ERROR_KERNEL;

		cusparseStatus_t  cudaStat1=cusparseScsrmm(
						handle, CUSPARSE_OPERATION_NON_TRANSPOSE, ref_nb, 10, query_nb, 1.0f, descrA, rnkCUDAtr,
						csrRowPtrACUDA, csrColIndACUDA, B, query_nb, 0.0f, C, ref_nb );//C = alpha*op(A)*B+beta*C

		cudaEventRecord(stop_event, 0);
				cudaEventSynchronize(stop_event);


		if (cudaStat1 != CUSPARSE_STATUS_SUCCESS)
		{
			printf("ERROR: executing sparse multiplication\n");
			exit(2);
		}


		HANDLE_ERROR_KERNEL;
		HANDLE_ERROR(cudaFree(B));
		HANDLE_ERROR(cudaFree(C));
		HANDLE_ERROR(cudaFree(csrColIndACUDA));
		//return; 
		 */			
		//-----------------------------------------------------------------------------

		//--------------------------------using our own kernel----------------------------

		HANDLE_ERROR(cudaMemset(X_k,0,sizeof(float)*ref_nb*3));
		HANDLE_ERROR(cudaMemset(S_k,0,sizeof(float)*ref_nb*6));
		HANDLE_ERROR(cudaMemset(N_k,0,sizeof(float)*ref_nb));
		GMEMcomputeXkNkSkKernelTr<<<ref_nb,MAX_THREADS>>>(indCUDAtr,queryCUDA,rnkIndCUDA,csrRowPtrACUDA,X_k,N_k,S_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;

		//-------------------------------------------------------------------------------


		//too long of a kernel->it is kind of unestable and we don't seem to gain much
		//GMEMcomputeXkNkSkKernelNoAtomic<<<ref_nb,MAX_THREADS_CUDA>>>(indCUDA,queryCUDA,rnkCUDA,X_k,N_k,S_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;//it does not require X_k normalization
		//-------------------------debug----------------------------
		/*
		if(frame==20)
		{
		printf("========================DEBUGGING: S_k on CUDA. IterEM=%d\n=====================",numIterEM);
		float *N_khost=new float[ref_nb];
		float *X_khost=new float[ref_nb*3];
		float *S_khost=new float[ref_nb*6];
		HANDLE_ERROR( cudaMemcpy( X_khost, X_k, sizeof(float) * ref_nb *3, cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( N_khost, N_k, sizeof(float) * ref_nb *1, cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( S_khost, S_k, sizeof(float) * ref_nb *6, cudaMemcpyDeviceToHost ) );
		printf("N_k=[\n");
		for(int ii=0;ii<ref_nb;ii++)
		{
			printf("%g ",N_khost[ii]);
		}
		printf("];\n");

		delete[] N_khost;
		delete[] X_khost;
		delete[] S_khost;
		}
		 */
		//--------------------------------------------------------------------------------


		//printGaussianMixtureModelCUDA(vecGMCUDA,refTempCUDA,0);
		GMEMupdateGaussianParametersKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,X_k,S_k,N_k,ref_nb);HANDLE_ERROR_KERNEL;
		if( regularizePrecisionMatrixConstants::lambdaMin < 0 )
		{
			printf("ERROR: before GMEMregularizeWkKernel: regularizePrecisionMatrixConstants were not set\n");
			exit(3);
		}
		GMEMregularizeWkKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,ref_nb, W4DOF, regularizePrecisionMatrixConstants::lambdaMin, regularizePrecisionMatrixConstants::lambdaMax, regularizePrecisionMatrixConstants::maxExcentricity);HANDLE_ERROR_KERNEL;

		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);HANDLE_ERROR_KERNEL;
		GMEMcheckDeadCellsKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,N_k,ref_nb,totalAlpha);HANDLE_ERROR_KERNEL;

		//------------------------------------debug vecGM-------------------------------------------
		/*
		printf("DEBUGGING: vecGM on CUDA\n");
		HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
		printf("[\n");
		for(int ii=0;ii<ref_nb;ii++)
		{
			printf("%g ",vecGMtemp[ii].alpha_k);
		}
		printf("];\n");
		exit(2);
		 */
		//-----------------------------------------------------------------------------------------


		//calculate log likelihood
		llOld3=llOld2;
		llOld2=llOld;
		llOld=ll;
		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);
		ll=addTotalLikelihood(imgDataCUDA,likelihoodVecCUDA,finalSumVectorInHostF,vecGMCUDA,indCUDA,queryCUDA,query_nb,ref_nb,totalAlpha);

		//printf("Frame=%d;iter=%d;Log-likelihood =%16.12f\n",frame,numIterEM,ll);
		numIterEM++;


		//------------------------debug: print out iterations----------------------------
		/*
		if(frame==20)
		{
			//copy results to vecGMtemp
			HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );

			printf("DEBUGGING: printing out frame by frame\n");
			char buffer[128];
			sprintf(buffer,"%.4d",numIterEM-1);
			string itoa=string(buffer);

			char buffer2[128];
			sprintf(buffer2,"%d",frame);
			string itoaFrame=string(buffer2);

			if (numIterEM==2)
			{
				string cmd("mkdir " + debugPath + itoaFrame +"/");
				int error=system(cmd.c_str());
			}

			ofstream outXML((debugPath + itoaFrame +"/debugEMGM_iter"+ itoa +".xml").c_str());

			GaussianMixtureModelCUDA::writeXMLheader(outXML);
			for(unsigned int ii=0;ii<ref_nb;ii++) vecGMtemp[ii].writeXML(outXML,ii);
			GaussianMixtureModelCUDA::writeXMLfooter(outXML);
			outXML.close();


			//------------------save Rnk and Ind
			float *rnkHost=new float[maxGaussiansPerVoxel*query_nb];
			int *indHost=new int[maxGaussiansPerVoxel*query_nb];
			float2 *rnkHost2=(float2*)malloc(maxGaussiansPerVoxel*query_nb*sizeof(float2));
			HANDLE_ERROR( cudaMemcpy( rnkHost, rnkCUDA, sizeof(float) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
			HANDLE_ERROR( cudaMemcpy( indHost, indCUDA, sizeof(int) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
			HANDLE_ERROR( cudaMemcpy( rnkHost2, rnkIndCUDA, sizeof(float2) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );

			FILE *pFile=fopen((debugPath + itoaFrame +"/rnk_iter"+ itoa +".txt").c_str(),"w");
			for(int ii=0;ii<query_nb;ii++)
			{
				for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
				{
					fprintf(pFile," %g",rnkHost[ii+query_nb*jj]);
				}
				fprintf(pFile,";\n");
			}		
			fclose(pFile);
			pFile=fopen((debugPath + itoaFrame +"/ind_iter"+ itoa +".txt").c_str(),"w");
			for(int ii=0;ii<query_nb;ii++)
			{
				for(int jj=0;jj<maxGaussiansPerVoxel;jj++)
				{
					fprintf(pFile," %d",indHost[ii+query_nb*jj]);
				}
				fprintf(pFile,";\n");
			}		
			fclose(pFile);
			HANDLE_ERROR( cudaMemcpy( indHost, indCUDAtr, sizeof(int) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
			pFile=fopen((debugPath + itoaFrame +"/indTr_iter"+ itoa +".txt").c_str(),"w");
			for(int ii=0;ii<query_nb*maxGaussiansPerVoxel;ii++)
			{
					fprintf(pFile," %d",indHost[ii]);
			}
			fprintf(pFile,";\n");
			fclose(pFile);
			pFile=fopen((debugPath + itoaFrame +"/rnkTr_iter"+ itoa +".txt").c_str(),"w");
			for(int ii=0;ii<query_nb*maxGaussiansPerVoxel;ii++)
			{
				fprintf(pFile," %g",rnkHost2[ii].x);
			}
			fprintf(pFile,";\n");
			fclose(pFile);


			pFile=fopen((debugPath + itoaFrame +"/csrRowPtr_iter"+ itoa +".txt").c_str(),"w");
			for(int ii=0;ii<ref_nb+1;ii++)
			{			
				fprintf(pFile," %d",csrRowPtrAHost[ii]);					
			}		
			fprintf(pFile,";\n");

			delete[] rnkHost;
			delete[] indHost;
			free(rnkHost2);
		}
		 */
		//-------------------------------------------------------------------------------

	}//end of for(iter=...)
	printf("Frame=%d;iter=%d;Log-likelihood =%16.12f\n",frame,numIterEM,ll);

	//calculate split score
	//GMEMcalculateLocalKullbackDiversityKernel<<<ref_nb,MAX_THREADS_CUDA>>>(indCUDA,queryCUDA,rnkCUDA,N_k,vecGMCUDA,query_nb,ref_nb);HANDLE_ERROR_KERNEL;

	//NOT needed anymore since we use classifier for split score. TODO: implement feature calculation in GPU
	//GMEMcalculateLocalKullbackDiversityKernelTr<<<ref_nb,MAX_THREADS_CUDA>>>(queryCUDA,rnkIndCUDA,N_k,csrRowPtrACUDA,vecGMCUDA,query_nb,ref_nb);HANDLE_ERROR_KERNEL;
	//---------------------debug-------------------------------
	/*
		HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
		printf("[;\n");
		for(int ii=0;ii<ref_nb;ii++) printf("%g ",vecGMtemp[ii].splitScore);
		printf("];\n");
	 */
	//----------------------


	//copy results to vecGMtemp
	HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );



	//release device memory
	HANDLE_ERROR( cudaFree( vecGMCUDA ) );
	HANDLE_ERROR( cudaFree( totalAlphaTempCUDA ) );
	//HANDLE_ERROR( cudaFree(indCUDA ) );
	HANDLE_ERROR( cudaFree(refTempCUDA ) );
	HANDLE_ERROR( cudaFree(S_k ) );
	HANDLE_ERROR( cudaFree(N_k ) );
	HANDLE_ERROR( cudaFree(X_k ) );
	//( cudaFree(likelihoodTempCUDA ) );
	HANDLE_ERROR( cudaFree(likelihoodVecCUDA ) );
	HANDLE_ERROR(cudaFree(csrRowPtrACUDA));
	HANDLE_ERROR(cudaFree(rnkIndCUDA));

	cusparseDestroy(handle);
	cusparseDestroyMatDescr(descrA);

	//release host memory
	free(finalSumVectorInHostD);
	free(finalSumVectorInHostF);
	free(csrRowPtrAHost);

	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
	printf(" done in %f secs for %d EM iterations (%f secs per iteration)\n", elapsedTime/1000, numIterEM-1, elapsedTime/((numIterEM-1)*1000));

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
}


/*
__device__ GaussianMixtureModelCUDA::GaussianMixtureModelCUDA()
{
	supervoxelNum = 0;
}
*/

//=================================================================================
//main interface to run variational inference in GPU with supervoxels, so the same assignment is guaranteed for voxels withihn the same supervoxel
void GMEMvariationalInferenceCUDAWithSupervoxels(float *queryCUDA,float *imgDataCUDA,pxi *rnkCUDA,pxi *rnkCUDAtr,int *indCUDA,int *indCUDAtr,float *centroidLabelPositionCUDA,long long int *labelListPtrCUDA,GaussianMixtureModelCUDA *vecGMtemp,long long int query_nb,int ref_nb,unsigned short int numLabels,int maxIterEM,double tolLikelihood,int devCUDA,int frame, bool W4DOF, string debugPath)
{
	if(query_nb>((long long int)MAX_THREADS) *((long long int)MAX_BLOCKS))
	{
		printf("ERROR: Too many query points for the GPU resources\n");//knnCUDA does not have while(tid<ref_nb) to handle infinite query_nb because of syncthreads
		exit(0);
	}
	if(ref_nb>MAX_BLOCKS)
	{
		printf("ERROR: Too many reference points for the GPU resources\n");
		exit(0);
	}

	//host variables
	double *finalSumVectorInHostD;
	float *finalSumVectorInHostF;
	int *csrRowPtrAHost;

	//CUDA variables 
	//pxi *rnkCUDA=NULL;
	//int *indCUDA=NULL;
	GaussianMixtureModelCUDA* vecGMCUDA=NULL;
	double *totalAlphaTempCUDA=NULL;
	float *refTempCUDA=NULL;
	float *X_k=NULL;//maximization step
	float *N_k=NULL;//maximization step
	float *S_k=NULL;//maximization step
	//float *likelihoodTempCUDA=NULL;
	float *likelihoodVecCUDA=NULL;
	int *csrRowPtrACUDA;
	float2 *rnkIndCUDA=NULL;


	//cusparse matrix
	cusparseHandle_t handle=0;
	cusparseMatDescr_t descrA=0;
	initializeCuSparseLinrary(handle,descrA);

	//calculate numThreads and numGrids for different kernels
	int numThreads_ref=std::min(MAX_THREADS,ref_nb);
	int numGrids_ref=std::min(MAX_BLOCKS,(ref_nb+numThreads_ref-1)/numThreads_ref);
long long 	int numThreads_query=std::min((long long int)MAX_THREADS,query_nb);
long long 	int numGrids_query=std::min((long long int)MAX_BLOCKS,(query_nb+numThreads_query-1)/numThreads_query);
long long 	int numThreads_labels=std::min((long long int)MAX_THREADS,(long long int)numLabels);
long long 	int numGrids_labels=std::min((long long int)MAX_BLOCKS,(long long int)((numLabels+numThreads_labels-1)/numThreads_labels));

	int sizeLikelihoodVec=(int)pow(2.0f,(int)ceil(log2((float)numLabels)));//we need it to be apower of 2 in order to perform summation as a kernel (dot product example)
long long 	int numThreads_likelihood=std::min((long long int)MAX_THREADS,(long long int)sizeLikelihoodVec);
long long 	int numGrids_likelihood=std::min((long long int)MAX_BLOCKS,(long long int)((sizeLikelihoodVec+numThreads_likelihood-1)/numThreads_likelihood));

	//select GPU and check maximum meory available and check that we have enough memory
	size_t memoryNeededInBytes=ref_nb*sizeof(GaussianMixtureModelCUDA)+
			numGrids_ref*sizeof(double)+
			ref_nb*dimsImage*sizeof(float)+
			ref_nb*(1+dimsImage+dimsImage*(1+dimsImage)/2)*sizeof(float)+
			numGrids_query*sizeof(float)+
			sizeof(int)*(ref_nb+1)+
			sizeof(float2)*query_nb*maxGaussiansPerVoxel;//this is the main load aside from preallocated rnkCUDA,rnkCUDAtr and indCUDA,indCUDAtr

	CUcontext cuContext;
	CUdevice  cuDevice=devCUDA;
	cuCtxCreate(&cuContext, 0, cuDevice);
	size_t memTotal,memAvailable;
	cuMemGetInfo(&memAvailable,&memTotal);
#if defined(_WIN32) || defined(_WIN64)
	printf( "Memory required: %lu;CUDA choosing device:  %Iu;memory available=%Iu;total mem=%Iu\n", memoryNeededInBytes,devCUDA,memAvailable,memTotal);
#else
	printf( "Memory required: %lu;CUDA choosing device:  %zu;memory available=%zu;total mem=%zu\n", memoryNeededInBytes,devCUDA,memAvailable,memTotal);
#endif
	cuCtxDetach (cuContext);

	/*for some reason this reads wrong in TEsla
	if(memAvailable<memoryNeededInBytes)
	{
		printf("ERROR: not enough memory in GPU!!\n");
		exit(2);
	}
	*/
	// capture the start time
	cudaEvent_t     start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	//allocate memory in host
	finalSumVectorInHostD=(double*)malloc(sizeof(double)*numGrids_ref);
	finalSumVectorInHostF=(float*)malloc(sizeof(float)*numGrids_likelihood);
	csrRowPtrAHost=(int *)malloc(sizeof(int)*(ref_nb+1));

	//allocate memory in device
	HANDLE_ERROR( cudaMalloc( (void**)&(vecGMCUDA), ref_nb*sizeof(GaussianMixtureModelCUDA) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(totalAlphaTempCUDA), numGrids_ref*sizeof(double) ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&indCUDA, query_nb*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&refTempCUDA, ref_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&N_k, ref_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&X_k, ref_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&S_k, ref_nb*dimsImage*(dimsImage+1)*sizeof(float)/2 ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&likelihoodTempCUDA, query_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&likelihoodVecCUDA, sizeLikelihoodVec*sizeof(float) ) );
	HANDLE_ERROR(cudaMalloc((void**)&csrRowPtrACUDA,sizeof(int)*(ref_nb+1)));
	HANDLE_ERROR(cudaMalloc((void**)&rnkIndCUDA,sizeof(float2)*numLabels*maxGaussiansPerVoxel));

	//wrap pointers for thurst library
	thrust::device_ptr<int> thrust_indCUDAtr(indCUDAtr);//keys
	thrust::device_ptr<float2> thrust_rnkIndCUDA(rnkIndCUDA);//values

	//upload vecGM info
	HANDLE_ERROR( cudaMemcpy( vecGMCUDA, vecGMtemp, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyHostToDevice ) );

	//iterate variational inference for one image
	int numIterEM=1;
	double totalAlpha;
	double ll=-1.0,llOld=ll*10,llOld2=ll*100,llOld3=ll*1000;//sometimes it oscillates between multiple values
	while(fabs(ll-llOld)/fabs(llOld)>tolLikelihood && fabs(ll-llOld2)/fabs(llOld2)>tolLikelihood && fabs(ll-llOld3)/fabs(llOld3)>tolLikelihood && numIterEM<=maxIterEM)
	{
		
		//------------------------debug each iteration and save it in a file (look at the function to change the destination folder)
		//HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
		//writeXMLdebugCUDA(vecGMtemp,ref_nb);
		//---------------------------------------------------------------------------------------------------------
		
		//calculate nearest neighbors
		GMEMcopyGaussianCenter2ConstantMemoryKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,refTempCUDA,ref_nb);HANDLE_ERROR_KERNEL;
		
		knnCUDAinPlace(indCUDA,centroidLabelPositionCUDA,refTempCUDA,numLabels,ref_nb);//calculate the possible assignment using the centroids of each Gaussian

		//calculate responsibilities (expectation step)
		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);
		GMEMupdatePriorConstantsCUDA(vecGMCUDA,ref_nb,DiGamma(totalAlpha));

		
		GMEMcomputeRnkCUDAInplaceWithSupervoxels(rnkCUDA,indCUDA,queryCUDA,query_nb,numLabels,vecGMCUDA,labelListPtrCUDA);


		//---------------------------------------------------------------------------------------------------

		//---------------------------------------------------------------------------------
		//use thrust library to transpose Rnk matrix
		// wrap raw pointer with a device_ptr
		GMEMcopyRnkBeforeSortKernel<<<numGrids_labels,numThreads_labels>>>(rnkCUDA,rnkIndCUDA,numLabels);HANDLE_ERROR_KERNEL;

		//amatf
		//HANDLE_ERROR(cudaMemcpy(indCUDAtr,indCUDA,sizeof(int)*maxGaussiansPerVoxel*query_nb,cudaMemcpyDeviceToDevice));//because we need indCUDA later
		memCpyDeviceToDeviceKernel<<<numGrids_labels,numThreads_labels>>>(indCUDA,indCUDAtr,maxGaussiansPerVoxel*numLabels);HANDLE_ERROR_KERNEL;


		thrust::sort_by_key(thrust_indCUDAtr, thrust_indCUDAtr+maxGaussiansPerVoxel*numLabels, thrust_rnkIndCUDA);//super fast radix sorting (166e6 elements/sec in my GPU) 

		//set row compress index format
		GMEMsetcsrRowPtrA(csrRowPtrAHost,csrRowPtrACUDA,indCUDAtr,numLabels,ref_nb);


		
		//--------------------------------using our own kernel----------------------------

		HANDLE_ERROR(cudaMemset(X_k,0,sizeof(float)*ref_nb*3));
		HANDLE_ERROR(cudaMemset(S_k,0,sizeof(float)*ref_nb*6));
		HANDLE_ERROR(cudaMemset(N_k,0,sizeof(float)*ref_nb));
		GMEMcomputeXkNkSkKernelTrWithSupervoxels<<<ref_nb,MAX_THREADS>>>(imgDataCUDA,labelListPtrCUDA,indCUDAtr,queryCUDA,rnkIndCUDA,csrRowPtrACUDA,X_k,N_k,S_k,query_nb,ref_nb);HANDLE_ERROR_KERNEL;

		//-------------------------------------------------------------------------------		
		//printGaussianMixtureModelCUDA(vecGMCUDA,refTempCUDA,0);
		GMEMupdateGaussianParametersKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,X_k,S_k,N_k,ref_nb);HANDLE_ERROR_KERNEL;
		if( regularizePrecisionMatrixConstants::lambdaMin < 0 )
		{
			printf("ERROR: before GMEMregularizeWkKernel: regularizePrecisionMatrixConstants were not set\n");
			exit(3);
		}
		GMEMregularizeWkKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,ref_nb, W4DOF, regularizePrecisionMatrixConstants::lambdaMin, regularizePrecisionMatrixConstants::lambdaMax, regularizePrecisionMatrixConstants::maxExcentricity);HANDLE_ERROR_KERNEL;

		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);HANDLE_ERROR_KERNEL;
		GMEMcheckDeadCellsKernel<<<numGrids_ref,numThreads_ref>>>(vecGMCUDA,N_k,ref_nb,totalAlpha);HANDLE_ERROR_KERNEL;

		//------------------------------------debug vecGM-------------------------------------------
		/*
		printf("DEBUGGING: vecGM on CUDA\n");
		HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
		printf("[\n");
		for(int ii=0;ii<ref_nb;ii++)
		{
			printf("%g ",vecGMtemp[ii].alpha_k);
		}
		printf("];\n");
		exit(2);
		 */
		//-----------------------------------------------------------------------------------------


		//calculate log likelihood
		llOld3=llOld2;
		llOld2=llOld;
		llOld=ll;
		totalAlpha=calculateTotalAlpha(vecGMCUDA,totalAlphaTempCUDA,finalSumVectorInHostD,ref_nb,numGrids_ref);
		ll=addTotalLikelihoodWithSupervoxels(imgDataCUDA,likelihoodVecCUDA,finalSumVectorInHostF,vecGMCUDA,indCUDA,queryCUDA,labelListPtrCUDA,query_nb,numLabels,totalAlpha);

		//printf("Frame=%d;iter=%d;Log-likelihood =%16.12f\n",frame,numIterEM,ll);
		numIterEM++;



	}//end of for(iter=...)
	printf("Frame=%d;iter=%d;Log-likelihood =%16.12f\n",frame,numIterEM,ll);

	//calculate split score
	//GMEMcalculateLocalKullbackDiversityKernel<<<ref_nb,MAX_THREADS_CUDA>>>(indCUDA,queryCUDA,rnkCUDA,N_k,vecGMCUDA,query_nb,ref_nb);HANDLE_ERROR_KERNEL;

	//NOT needed anymore since we use classifier for split score. TODO: implement feature calculation in GPU
	//GMEMcalculateLocalKullbackDiversityKernelTr<<<ref_nb,MAX_THREADS_CUDA>>>(queryCUDA,rnkIndCUDA,N_k,csrRowPtrACUDA,vecGMCUDA,query_nb,ref_nb);HANDLE_ERROR_KERNEL;
	//---------------------debug-------------------------------
	/*
		HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
		printf("[;\n");
		for(int ii=0;ii<ref_nb;ii++) printf("%g ",vecGMtemp[ii].splitScore);
		printf("];\n");
	 */
	//----------------------

	//---------------------debug rnk and indcuda----------------------------
	/*
	//pxi *rnkCUDA,int *indCUDA
	printf("DEBUGGING: rnk and indCUDA values\n");
	pxi* rnkHostAux=new pxi[numLabels*maxGaussiansPerVoxel];
	int*  indHostAux=new int[numLabels*maxGaussiansPerVoxel];
	HANDLE_ERROR( cudaMemcpy( rnkHostAux, rnkCUDA, sizeof(pxi) * numLabels*maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( indHostAux, indCUDA, sizeof(int) * numLabels*maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );

	int K=8;//selects the Gaussian for which we want to print the results
	printf("Ind and rnk for %d-th mixture\n",K);
	for(int ii=0;ii<maxGaussiansPerVoxel;ii++)
	{
		printf("%d;%g\n",indHostAux[K+ii*numLabels],rnkHostAux[K+ii*numLabels]); 
	}

	delete[] rnkHostAux;
	delete[] indHostAux;
	exit(2);
	*/
	//--------------------------------------------------------------------

	//copy results to vecGMtemp
	HANDLE_ERROR( cudaMemcpy( vecGMtemp, vecGMCUDA, sizeof(GaussianMixtureModelCUDA) * ref_nb, cudaMemcpyDeviceToHost ) );
	for(unsigned int jj = 0;jj<ref_nb;jj++) //reset number of supervoxels
		vecGMtemp[jj].supervoxelNum = 0;  


	//save rnk assignment
	pxi* rnkHostAux=new pxi[numLabels*maxGaussiansPerVoxel];
	int*  indHostAux=new int[numLabels*maxGaussiansPerVoxel];
	HANDLE_ERROR( cudaMemcpy( rnkHostAux, rnkCUDA, sizeof(pxi) * numLabels*maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( indHostAux, indCUDA, sizeof(int) * numLabels*maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );

	//copy results to vecGM		
	long long int pos = 0;
	int auxN;
	for(int ii = 0;ii<maxGaussiansPerVoxel;ii++)
	{
		for(unsigned int jj = 0;jj<numLabels;jj++)
		{
			if( rnkHostAux[pos] > 0.5 )//clear assignment
			{
				auxN = vecGMtemp[ indHostAux[pos] ].supervoxelNum;
				if( auxN < MAX_SUPERVOXELS_PER_GAUSSIAN )
				{
					vecGMtemp[ indHostAux[pos] ].supervoxelIdx[ vecGMtemp[ indHostAux[pos] ].supervoxelNum++ ] = jj;
				}
				else{
					//cout<<"WARNING: at GMEMvariationalInferenceCUDAWithSupervoxels: number of supervoxels per Gaussian exceeds limit. Most likely it is a Gaussian colecting background. You can increase MAX_SUPERVOXELS_PER_GAUSSIAN (currently it is "<<MAX_SUPERVOXELS_PER_GAUSSIAN<<") and recompile code. jj = "<<indHostAux[pos]<<";number of supervoxels="<<vecGMtemp[ indHostAux[pos] ].supervoxelNum<<endl;
				}
			}
			pos++;
		}
	}

	/*
	for(unsigned int jj = 0;jj<ref_nb;jj++) 
		if( vecGMtemp[jj].supervoxelNum > MAX_SUPERVOXELS_PER_GAUSSIAN )
		{
			cout<<"ERROR: at GMEMvariationalInferenceCUDAWithSupervoxels: number of supervoxels per Gaussian exceeds limit. Increase MAX_SUPERVOXELS_PER_GAUSSIAN and recompile code. jj = "<<jj<<";number of supervoxels="<<vecGMtemp[jj].supervoxelNum<<endl;
			exit(3);
		}
	*/

	//save rnk and ind if necessary
	if(strcmp(debugPath.c_str(),"")!=0)
	{
		
		FILE* fid=fopen(debugPath.c_str(),"wb");
		if(fid==NULL)
		{
			printf("ERROR: opening file %s to save rnk values from variational inference\n",debugPath.c_str());
			exit(2);
		}

		int numLabelsAux=(int)numLabels;
		fwrite(&numLabelsAux,sizeof(int),1,fid);//write number of labels
		fwrite(&maxGaussiansPerVoxel,sizeof(int),1,fid);//write maxGaussians per voxel
		fwrite(rnkHostAux,sizeof(pxi),numLabels*maxGaussiansPerVoxel,fid);//writing rnk
		fwrite(indHostAux,sizeof(int),numLabels*maxGaussiansPerVoxel,fid);//writing ind
		fclose(fid);

	}

	delete[] rnkHostAux;
	delete[] indHostAux;

	//release device memory
	HANDLE_ERROR( cudaFree( vecGMCUDA ) );
	HANDLE_ERROR( cudaFree( totalAlphaTempCUDA ) );
	//HANDLE_ERROR( cudaFree(indCUDA ) );
	HANDLE_ERROR( cudaFree(refTempCUDA ) );
	HANDLE_ERROR( cudaFree(S_k ) );
	HANDLE_ERROR( cudaFree(N_k ) );
	HANDLE_ERROR( cudaFree(X_k ) );
	//( cudaFree(likelihoodTempCUDA ) );
	HANDLE_ERROR( cudaFree(likelihoodVecCUDA ) );
	HANDLE_ERROR(cudaFree(csrRowPtrACUDA));
	HANDLE_ERROR(cudaFree(rnkIndCUDA));

	cusparseDestroy(handle);
	cusparseDestroyMatDescr(descrA);

	//release host memory
	free(finalSumVectorInHostD);
	free(finalSumVectorInHostF);
	free(csrRowPtrAHost);

	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,start, stop ) );
	printf(" done in %f secs for %d EM iterations (%f secs per iteration)\n", elapsedTime/1000, numIterEM-1, elapsedTime/((numIterEM-1)*1000));

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );
}



//==================================================================================================
void mainTestGMEMcomputeVariationalInference(long long int query_nb)
{
	//long long int query_nb=50000;
	int ref_nb=1200;
	int maxIterEM=10;
	double tolLikelihoodEM=1e-9;

	int dev=0;
	HANDLE_ERROR( cudaSetDevice( dev ) );

	//host variables
	float *query;
	GaussianMixtureModelCUDA *vecGM;
	float *imgData;
	float scale[dimsImage]={1.0,1.0,1.0};

	query  = (float *) malloc(query_nb * dimsImage * sizeof(float));
	vecGM = (GaussianMixtureModelCUDA*)malloc(ref_nb*sizeof(GaussianMixtureModelCUDA));
	imgData  = (float *) malloc(query_nb * sizeof(float));

	//generate random points
	// Init
	srand((unsigned int)time(NULL));
	for (int i=0 ; i<query_nb * dimsImage ; i++) query[i]  = 10*(float)rand() / (float)RAND_MAX;
	for (int i=0 ; i<query_nb  ; i++) imgData[i]  = (float)rand() / (float)RAND_MAX;

	for(int i=0;i<ref_nb;i++)
	{
		vecGM[i].beta_k=0.1;
		vecGM[i].nu_k=dimsImage+1.0;
		vecGM[i].alpha_k=(((float)rand() / (float)RAND_MAX))*100.0;
		for(int j=0;j<dimsImage;j++) vecGM[i].m_k[j]=(float)rand() / (float)RAND_MAX;
		vecGM[i].expectedLogDetCovarianceCUDA=-(float)rand() / (float)RAND_MAX;
		vecGM[i].expectedLogResponsivityCUDA=-(float)rand() / (float)RAND_MAX;

		memset(vecGM[i].W_k,0,dimsImage*(dimsImage+1)*sizeof(double)/2);
		vecGM[i].W_k[0]=0.33;//diagonal
		vecGM[i].W_k[3]=0.1;
		vecGM[i].W_k[5]=0.2;
		vecGM[i].W_k[1]=0.01;
		vecGM[i].W_k[2]=0.02;
		vecGM[i].W_k[4]=0.04;

		vecGM[i].beta_o=0.5;
		vecGM[i].nu_o=dimsImage+1.0;
		vecGM[i].alpha_o=(((float)rand() / (float)RAND_MAX)-0.5)*100.0;
		for(int j=0;j<dimsImage;j++) vecGM[i].m_o[j]=(float)rand() / (float)RAND_MAX;


		memset(vecGM[i].W_o,0,dimsImage*(dimsImage+1)*sizeof(double)/2);
		vecGM[i].W_o[0]=0.12;//diagonal
		vecGM[i].W_o[3]=0.15;
		vecGM[i].W_o[5]=0.17;

		vecGM[i].fixed=false;
	}

	//CUDA variables
	float *queryCUDA=NULL;
	float *imgDataCUDA=NULL;
	float *rnkCUDA=NULL;
	float *rnkCUDAtr=NULL;
	int *indCUDA=NULL;
	int *indCUDAtr=NULL;



	//allocate memory in device
	HANDLE_ERROR( cudaMalloc( (void**)&queryCUDA, query_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&imgDataCUDA, query_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&rnkCUDA, query_nb*maxGaussiansPerVoxel*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&rnkCUDAtr, query_nb*maxGaussiansPerVoxel*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&indCUDA, query_nb*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&indCUDAtr, query_nb*maxGaussiansPerVoxel*sizeof(int) ) );

	//copy initial elements
	HANDLE_ERROR( cudaMemcpy( imgDataCUDA, imgData, sizeof(float) * query_nb, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( queryCUDA, query, sizeof(float) * query_nb*dimsImage, cudaMemcpyHostToDevice ) );
	uploadScaleCUDA(scale);

	//main calculation
	GMEMvariationalInferenceCUDA(queryCUDA,imgDataCUDA,rnkCUDA,rnkCUDAtr,indCUDA,indCUDAtr,vecGM,query_nb,ref_nb,maxIterEM,tolLikelihoodEM,dev,0, true);

	//print out result
	/*
	for(int ii=0;ii<ref_nb;ii++)
	{
		printf("%g \n",vecGM[ii].alpha_k);
	}
	 */
	//release device memory
	HANDLE_ERROR(cudaFree(imgDataCUDA));
	HANDLE_ERROR(cudaFree(queryCUDA));
	HANDLE_ERROR(cudaFree(indCUDA));
	HANDLE_ERROR(cudaFree(indCUDAtr));
	HANDLE_ERROR(cudaFree(rnkCUDA));
	HANDLE_ERROR(cudaFree(rnkCUDAtr));

	//release host memory
	free(query);
	free(vecGM);
	free(imgData);
}

void GMEMinitializeMemory(float **queryCUDA,float *query,float **imgDataCUDA,float *imgData,float *scale,long long int query_nb,float **rnkCUDA,int **indCUDA,int **indCUDAtr)
{
	//select GPU and check maximum meory available and check that we have enough memory
	/*
	size_t memoryNeededInBytes=query_nb*(dimsImage+1+maxGaussiansPerVoxel*3)*sizeof(float);

	CUcontext cuContext;
	CUdevice  cuDevice=devCUDA;
	cuCtxCreate(&cuContext, 0, cuDevice);
	size_t memTotal,memAvailable;
	cuMemGetInfo(&memAvailable,&memTotal);
	printf( "Memory required: %lu;CUDA choosing device:  %d;memory available=%lu;total mem=%lu\n", memoryNeededInBytes,devCUDA,memAvailable,memTotal);
	cuCtxDetach (cuContext);
	 */


	//allocate memory in device
	HANDLE_ERROR( cudaMalloc( (void**)&(*queryCUDA), query_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*imgDataCUDA), query_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*rnkCUDA), query_nb*maxGaussiansPerVoxel*sizeof(float) ) );
	//HANDLE_ERROR( cudaMalloc( (void**)&(*rnkCUDAtr), query_nb*maxGaussiansPerVoxel*sizeof(float) ) ); Only needed if we use Csparse
	HANDLE_ERROR( cudaMalloc( (void**)&(*indCUDA), query_nb*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*indCUDAtr), query_nb*maxGaussiansPerVoxel*sizeof(int) ) );


	//copy initial elements
	HANDLE_ERROR( cudaMemcpy( *imgDataCUDA, imgData, sizeof(float) * query_nb, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( *queryCUDA, query, sizeof(float) * query_nb*dimsImage, cudaMemcpyHostToDevice ) );
	uploadScaleCUDA(scale);
}
void GMEMinitializeMemoryWithSupervoxels(float **queryCUDA,float *query,float **imgDataCUDA,float *imgData,float *scale,long long int query_nb,float **rnkCUDA,int **indCUDA,int **indCUDAtr,unsigned short int numLabels,float **centroidLabelPositionCUDA,float* centroidLabelPosition,long long int **labelListPtrCUDA,long long int *labelListPtrHOST)
{
	//select GPU and check maximum meory available and check that we have enough memory
	/*
	size_t memoryNeededInBytes=query_nb*(dimsImage+1+maxGaussiansPerVoxel*3)*sizeof(float);

	CUcontext cuContext;
	CUdevice  cuDevice=devCUDA;
	cuCtxCreate(&cuContext, 0, cuDevice);
	size_t memTotal,memAvailable;
	cuMemGetInfo(&memAvailable,&memTotal);
	printf( "Memory required: %lu;CUDA choosing device:  %d;memory available=%lu;total mem=%lu\n", memoryNeededInBytes,devCUDA,memAvailable,memTotal);
	cuCtxDetach (cuContext);
	 */


	//allocate memory in device
	HANDLE_ERROR( cudaMalloc( (void**)&(*queryCUDA), query_nb*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*imgDataCUDA), query_nb*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*rnkCUDA), numLabels*maxGaussiansPerVoxel*sizeof(float) ) );//numLabels<<query_nb -> we save a lot of memory
	//HANDLE_ERROR( cudaMalloc( (void**)&(*rnkCUDAtr), query_nb*maxGaussiansPerVoxel*sizeof(float) ) ); Only needed if we use Csparse
	HANDLE_ERROR( cudaMalloc( (void**)&(*indCUDA), numLabels*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*indCUDAtr), numLabels*maxGaussiansPerVoxel*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*centroidLabelPositionCUDA), numLabels*dimsImage*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(*labelListPtrCUDA), (numLabels+1)*sizeof(long long int) ) );
	

	//copy initial elements
	HANDLE_ERROR( cudaMemcpy( *imgDataCUDA, imgData, sizeof(float) * query_nb, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( *queryCUDA, query, sizeof(float) * query_nb*dimsImage, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( *centroidLabelPositionCUDA, centroidLabelPosition, sizeof(float) * numLabels*dimsImage, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( *labelListPtrCUDA, labelListPtrHOST, sizeof(long long int) * (numLabels+1), cudaMemcpyHostToDevice ) );
	uploadScaleCUDA(scale);
}
void GMEMreleaseMemory(float **queryCUDA,float **imgDataCUDA,float **rnkCUDA,int **indCUDA,int **indCUDAtr)
{
	//release device memory
	HANDLE_ERROR(cudaFree(*imgDataCUDA));
	(*imgDataCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*queryCUDA));
	(*queryCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*rnkCUDA));
	(*rnkCUDA)=NULL;
	//HANDLE_ERROR(cudaFree(*rnkCUDAtr));
	//(*rnkCUDAtr)=NULL;
	HANDLE_ERROR(cudaFree(*indCUDA));
	(*indCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*indCUDAtr));
	(*indCUDAtr)=NULL;
}

void GMEMreleaseMemoryWithSupervoxels(float **queryCUDA,float **imgDataCUDA,float **rnkCUDA,int **indCUDA,int **indCUDAtr,float **centroidLabelPositionCUDA,long long int **labelListPtrCUDA)
{
	//release device memory
	HANDLE_ERROR(cudaFree(*imgDataCUDA));
	(*imgDataCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*queryCUDA));
	(*queryCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*rnkCUDA));
	(*rnkCUDA)=NULL;
	//HANDLE_ERROR(cudaFree(*rnkCUDAtr));
	//(*rnkCUDAtr)=NULL;
	HANDLE_ERROR(cudaFree(*indCUDA));
	(*indCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*indCUDAtr));
	(*indCUDAtr)=NULL;
	HANDLE_ERROR(cudaFree(*centroidLabelPositionCUDA));
	(*centroidLabelPositionCUDA)=NULL;
	HANDLE_ERROR(cudaFree(*labelListPtrCUDA));
	(*labelListPtrCUDA)=NULL;
}

//=========================================================================================
ostream& GaussianMixtureModelCUDA::writeXML(ostream& os,int id)
{
	os<<"<GaussianMixtureModel ";;
	os<<"id=\""<<id<<"\" lineage=\""<<0<<"\" parent=\""<<0<<"\" dims=\""<<dimsImage<<"\" splitScore=\""<<splitScore<<"\"";

	os<<" scale=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<1.0f<<" ";
	os<<"\""<<endl;

	//write variables values
	os<<"nu=\""<<nu_k<<"\" beta=\""<<beta_k<<"\" alpha=\""<<alpha_k<<"\"";
	os<<" m=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<m_k[ii]<<" ";

	//for dimsImage==3
	os<<"\" W=\""<<W_k[0]<< " "<<W_k[1]<< " "<<W_k[2]<< " "<<W_k[1]<< " "<<W_k[3]<< " "<<W_k[4]<< " "<<W_k[2]<< " "<<W_k[4]<< " "<<W_k[5];

	os<<"\""<<endl;


	//write priors values
	os<<"nuPrior=\""<<nu_o<<"\" betaPrior=\""<<beta_o<<"\" alphaPrior=\""<<alpha_o<<"\" distMRFPrior=\""<<0.0<<"\"";
	os<<" mPrior=\"";
	for(int ii=0;ii<dimsImage;ii++) os<<m_o[ii]<<" ";
	os<<"\" WPrior=\""<<W_o[0]<< " "<<W_o[1]<< " "<<W_o[2]<< " "<<W_o[1]<< " "<<W_o[3]<< " "<<W_o[4]<< " "<<W_o[2]<< " "<<W_o[4]<< " "<<W_o[5];

	os<<"\">"<<endl;


	os<<"</GaussianMixtureModel>"<<endl;

	return os;

}

//===================================================================================
void findVoxelAssignmentBasedOnRnk(int *kAssignment,long long int query_nb,float *rnkCUDA,int *indCUDA)
{

	float *rnkHost=new float[maxGaussiansPerVoxel*query_nb];
	int *indHost=new int[maxGaussiansPerVoxel*query_nb];
	HANDLE_ERROR( cudaMemcpy( rnkHost, rnkCUDA, sizeof(float) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );
	HANDLE_ERROR( cudaMemcpy( indHost, indCUDA, sizeof(int) * query_nb *maxGaussiansPerVoxel, cudaMemcpyDeviceToHost ) );

	float auxNorm,aux,maxRnk;
	long long int offset;
	int K;
	for(long long int ii=0;ii<query_nb;ii++)
	{
		auxNorm=0.0f;
		offset=ii;
		maxRnk=0.0f;
		for(int kk=0;kk<maxGaussiansPerVoxel;kk++)
		{
			aux=rnkHost[offset];
			auxNorm+=aux;
			if(aux>maxRnk)
			{
				maxRnk=aux;
				K=indHost[offset];
			}
			offset+=query_nb;   
		}
		if(auxNorm>0.0)
		{
			if(maxRnk/auxNorm>0.3)//clear assignment
			{
				kAssignment[ii]=K;
			}else kAssignment[ii]=-1;//no clear assignment
		}else kAssignment[ii]=-1;//no assignment
	}

	delete []rnkHost;
	delete []indHost;
}

void copyScaleToDvice(float *scale)
{
	HANDLE_ERROR(cudaMemcpyToSymbol(scaleGMEMCUDA,scale, dimsImage * sizeof(float)));//constant memory
}


//======================================================
//copied from ostream& GaussianMixtureModel::writeXML(ostream& os) in order to be able to debug EM iterations withing the GPU
void writeXMLdebugCUDA(GaussianMixtureModelCUDA *vecGMtemp,unsigned int numElem)
{
	char logBuffer[32];
	sprintf(logBuffer,"%.4d",iterEMdebug++);//increment iter in order to save next iteration
	string itoaLog=string(logBuffer);
	string filename("/Users/amatf/TrackingNuclei/tmp/GMEMtracking3D_1320774859/XML_EMiter/XML_EMiter_frame127_TM"+itoaLog+".xml");//change this to save debug file
	float scale[dimsImage]={1.0f,1.0f,5.0f};
	cout<<"Saving iteration at "<<filename<<endl;
	ofstream os(filename.c_str());
	
	//write XML header 
	os<<"<?xml version=\"1.0\" encoding=\"utf-8\"?>"<<endl<<"<document>"<<endl;
	
	double *auxW=new double[dimsImage*dimsImage];
	int count=0;
	for(unsigned int rr=0;rr<numElem;rr++)
	{
		GaussianMixtureModelCUDA *GM=&(vecGMtemp[rr]);
		os<<"<GaussianMixtureModel ";;
		os<<"id=\""<<rr<<"\" lineage=\""<<0<<"\" parent=\""<<0<<"\" dims=\""<<dimsImage<<"\" splitScore=\""<<GM->splitScore<<"\"";

		os<<" scale=\"";
		for(int ii=0;ii<dimsImage;ii++) os<<scale[ii]<<" ";
		os<<"\""<<endl;

		//write variables values
		os<<"nu=\""<<GM->nu_k<<"\" beta=\""<<GM->beta_k<<"\" alpha=\""<<GM->alpha_k<<"\"";
		os<<" m=\"";
		for(int ii=0;ii<dimsImage;ii++) os<<GM->m_k[ii]<<" ";
		os<<"\" W=\"";
		
		count=0;
		for(int ii=0;ii<dimsImage;ii++)
		{
			auxW[ii*dimsImage+ii]=GM->W_k[count++];
			for(int jj=ii+1;jj<dimsImage;jj++)
			{
				auxW[ii*dimsImage+jj]=GM->W_k[count++];
				auxW[jj*dimsImage+ii]=auxW[ii*dimsImage+jj];
			}
		}
		for(int ii=0;ii<dimsImage*dimsImage;ii++) os<<auxW[ii]<<" ";
		os<<"\""<<endl;


		//write priors values
		os<<"nuPrior=\""<<GM->nu_o<<"\" betaPrior=\""<<GM->beta_o<<"\" alphaPrior=\""<<GM->alpha_o<<"\" distMRFPrior=\""<<0<<"\"";
		os<<" mPrior=\"";
		for(int ii=0;ii<dimsImage;ii++) os<<GM->m_o[ii]<<" ";
		os<<"\" WPrior=\"";
		count=0;
		for(int ii=0;ii<dimsImage;ii++)
		{
			auxW[ii*dimsImage+ii]=GM->W_o[count++];
			for(int jj=ii+1;jj<dimsImage;jj++)
			{
				auxW[ii*dimsImage+jj]=GM->W_o[count++];
				auxW[jj*dimsImage+ii]=auxW[ii*dimsImage+jj];
			}
		}
		for(int ii=0;ii<dimsImage*dimsImage;ii++) os<<auxW[ii]<<" ";
		os<<"\">"<<endl;
		os<<"</GaussianMixtureModel>"<<endl;
	}
	//write XML footer
	os<<"</document>"<<endl;
	
	os.close();

	delete[] auxW;
}
