#include "EllipticalHaarFeatures.h"
#include "book.h"
#include "cuda.h"
#include <iostream>
#include <math.h>
/* Includes for HealPix*/
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__device__ static const double PI_= 3.14159265358979311600;
__device__ static const double SQRT3_= 1.73205080756887719318;

__constant__  double r0CUDA[1];//initial radius (based on sigma) to define central cell ellipsoid
__constant__  double kSigmaCUDA[1];//factor to define outer rings dimensions in the C-HoG block (r_m=(r0+m*kSigma)*sigma)
__constant__  int numCellsHEALPixCUDA[1]; // number of sectors per fixed radius using HEALpix

__constant__ int x2pix[128];//necessary for HEALPIX routine
__constant__ int y2pix[128];

__device__ static const double piover2 = 1.5707963267948966; 
__device__ static const double twopi = 6.2831853071795862;
__device__ static const int    ns_max = 8192;
  

__constant__ float kernelSeparableConv[1+2*maxRadiusBox];//kernel for separable convolution. It does not make sense if it is larger than the box

texture<imageType, dimsImage, cudaReadModeElementType> textureImage; 


static const float constDoG_0 = 2.0f*sqrt(2.0f); //some ocnstants we need for DoG kernel
static const float constDoG_1 =(4.0f*sqrt(2.0f*log(2.0f)));

int basicEllipticalHaarFeatureVector::numCells;
int basicEllipticalHaarFeatureVector::numRings;




//=================================================================================================
//===================================HealPix function executed in GPU=============================
/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2010 Krzysztof M. Gorski, Eric Hivon,
 *                          Benjamin D. Wandelt, Anthony J. Banday, 
 *                          Matthias Bartelmann, 
 *                          Reza Ansari & Kenneth M. Ganga 
 *
 *
 *  This file is part of HEALPix.
 *
 *  HEALPix is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  HEALPix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with HEALPix; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix see http://healpix.jpl.nasa.gov
 *
 *----------------------------------------------------------------------------- */
/* vec2pix_nest.c */

/* Local Includes */
//#include "chealpix.h"
void mk_xy2pix(int *x2pix, int *y2pix) {
  /* =======================================================================
   * subroutine mk_xy2pix
   * =======================================================================
   * sets the array giving the number of the pixel lying in (x,y)
   * x and y are in {1,128}
   * the pixel number is in {0,128**2-1}
   *
   * if  i-1 = sum_p=0  b_p * 2^p
   * then ix = sum_p=0  b_p * 4^p
   * iy = 2*ix
   * ix + iy in {0, 128**2 -1}
   * =======================================================================
   */
  int i, K,IP,I,J,ID;
  
  for(i = 0; i < 127; i++) x2pix[i] = 0;
  for( I=1;I<=128;I++ ) {
    J  = I-1;//            !pixel numbers
    K  = 0;//
    IP = 1;//
    truc : if( J==0 ) {
      x2pix[I-1] = K;
      y2pix[I-1] = 2*K;
    }
    else {
      ID = (int)fmod((float)J,2.0f);
      J  = J/2;
      K  = IP*ID+K;
      IP = IP*4;
      goto truc;
    }
  }     
  
}
__device__ void vec2pix_nest( const long long int nside, double *vec, long long int *ipix) 
{

  /* =======================================================================
   * subroutine vec2pix_nest(nside, vec, ipix)
   * =======================================================================
   * gives the pixel number ipix (NESTED) corresponding to vector vec
   *
   * the computation is made to the highest resolution available (nside=8192)
   * and then degraded to that required (by integer division)
   * this doesn't cost more, and it makes sure that the treatement of round-off 
   * will be consistent for every resolution
   * =======================================================================
   */
  
  double z, za, z0, tt, tp, tmp, phi;
  int    face_num = 0,jp = 0,jm = 0;
  long long int   ifp = 0, ifm = 0;
  int    ix = 0, iy = 0, ix_low = 0, ix_hi = 0, iy_low = 0, iy_hi = 0, ipf = 0, ntt = 0;
  //double piover2 = 0.5*PI_, twopi = 2.0*PI_;
  //int    ns_max = 8192;
  //int x2pix[128], y2pix[128]; //set as __constant__ in the GPU since they are the same for all calls
  //static int x2pix[128], y2pix[128];
  //static char setup_done = 0;
  
  //if( nside<1 || nside>ns_max ) {
  //  fprintf(stderr, "%s (%d): nside out of range: %ld\n", __FILE__, __LINE__, nside);
  //  exit(0);
  //}

  //if( !setup_done ) {
    //mk_xy2pix(x2pix,y2pix);
    //setup_done = 1;
  //}
  
  z   = vec[2]/sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
  phi = 0.0;
  if (vec[0] != 0.0 || vec[1] != 0.0) {
    phi   = atan2(vec[1],vec[0]); /* in ]-pi, pi] */
    if (phi < 0.0) phi += twopi; /* in  [0, 2pi[ */
  }

  za = fabs(z);
  z0 = 2./3.;
  tt = phi / piover2; /* in [0,4[ */
  
  if( za<=z0 ) { /* equatorial region */
    
    /* (the index of edge lines increase when the longitude=phi goes up) */
    jp = (int)floor(ns_max*(0.5 + tt - z*0.75)); /* ascending edge line index */
    jm = (int)floor(ns_max*(0.5 + tt + z*0.75)); /* descending edge line index */
    
    /* finds the face */
    ifp = jp / ns_max; /* in {0,4} */
    ifm = jm / ns_max;
    
    if( ifp==ifm ) face_num = (int)fmod((float)ifp,4.0f) + 4; /* faces 4 to 7 */
    else if( ifp<ifm ) face_num = (int)fmod((float)ifp,4.0f); /* (half-)faces 0 to 3 */
    else face_num = (int)fmod((float)ifm,4.0f) + 8;           /* (half-)faces 8 to 11 */
    
    ix = (int)fmod((float)jm, (float)ns_max);
    iy = ns_max - (int)fmod((float)jp, (float)ns_max) - 1;
  }
  else { /* polar region, za > 2/3 */
    
    ntt = (int)floor(tt);
    if( ntt>=4 ) ntt = 3;
    tp = tt - ntt;
    tmp = sqrt( 3.*(1. - za) ); /* in ]0,1] */
    
    /* (the index of edge lines increase when distance from the closest pole
     * goes up)
     */
    /* line going toward the pole as phi increases */
    jp = (int)floor( ns_max * tp          * tmp ); 

    /* that one goes away of the closest pole */
    jm = (int)floor( ns_max * (1. - tp) * tmp );
    jp = (int)(jp < ns_max-1 ? jp : ns_max-1);
    jm = (int)(jm < ns_max-1 ? jm : ns_max-1);
    
    /* finds the face and pixel's (x,y) */
    if( z>=0 ) {
      face_num = ntt; /* in {0,3} */
      ix = ns_max - jm - 1;
      iy = ns_max - jp - 1;
    }
    else {
      face_num = ntt + 8; /* in {8,11} */
      ix =  jp;
      iy =  jm;
    }
  }
  
  ix_low = (int)fmod((float)ix,128.0f);
  ix_hi  =     ix/128;
  iy_low = (int)fmod((float)iy,128.0f);
  iy_hi  =     iy/128;


  ipf = (x2pix[ix_hi]+y2pix[iy_hi]) * (128 * 128)+ (x2pix[ix_low]+y2pix[iy_low]);
  ipf = (long long int)(ipf / pow((float)ns_max/nside,2));     /* in {0, nside**2 - 1} */
  *ipix =(long long int)( ipf + face_num*pow((float)nside,2)); /* in {0, 12*nside**2 - 1} */
  
}



//===============================================================================================


//=======================================================================================================
//===========================================================================
//eigen value functions: I need to have everythign in one file

//=============================================================================================
//=========================================================================
//determinant for 3x3 symmetric matrix
__device__  double determinantSymmetricW_3D(const double *W_k)
{
	return W_k[0]*(W_k[3]*W_k[5]-W_k[4]*W_k[4])-W_k[1]*(W_k[1]*W_k[5]-W_k[2]*W_k[4])+W_k[2]*(W_k[1]*W_k[4]-W_k[2]*W_k[3]);
}
//=========================================================================
//inverse for a 3x3 symmetric matrix
__device__  void inverseSymmetricW_3D(double *W,double *W_inverse)
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
//analytical solution for eigenvalues 3x3 real symmetric matrices
//formula for eigenvalues from http://en.wikipedia.org/wiki/Eigenvalue_algorithm#Eigenvalues_of_3.C3.973_matrices
__device__  void  eig3(const double *w, double *d, double *v)
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
	v[0]=w[1]*w[4]-w[2]*(w[3]-d[0]);v[1]=w[2]*w[1]-w[4]*(w[0]-d[0]);v[2]=(w[0]-d[0])*(w[3]-d[0])-w[1]*w[1];
	v[3]=w[1]*w[4]-w[2]*(w[3]-d[1]);v[4]=w[2]*w[1]-w[4]*(w[0]-d[1]);v[5]=(w[0]-d[1])*(w[3]-d[1])-w[1]*w[1];
	v[6]=w[1]*w[4]-w[2]*(w[3]-d[2]);v[7]=w[2]*w[1]-w[4]*(w[0]-d[2]);v[8]=(w[0]-d[2])*(w[3]-d[2])-w[1]*w[1];

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

	//adjust v in case zome eigenvalues are zeros
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
 __device__ void eig2(const double *w, double *d, double *v)
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


//===============================end of eigenvalus functionality============================================================

//===============================beginning of eigenvalues kernel============================================================
 //order indicates if we want to order the elements according to eigenvectors.
 //order=0 ->no need to order them; order<0 -> order in descend order;order>0->order in ascendant order
 __global__ void __launch_bounds__(MAX_THREADS_CUDA) computeEigenDecompositionKernel(double *W,int numEllipsoids,double *dCUDA,double* vCUDA, int order)
{
	double v[dimsImage*dimsImage];
	double d[dimsImage];
	double Wlocal[dimsImage*(dimsImage+1)/2];

	
	long long int tid=threadIdx.x + blockIdx.x * blockDim.x;
	long long int pos=tid;

	if(tid<numEllipsoids)
	{
		
		//copy from global memory
		pos=tid;
		for(int ii=0;ii<dimsImage*(dimsImage+1)/2;ii++)
		{
			Wlocal[ii] = W[pos];
			pos+=numEllipsoids;
		}
		if(dimsImage == 3)
			eig3(Wlocal, d, v);
		else if (dimsImage == 2)
			eig2(Wlocal, d, v);

		//order according to eigenvalues using bubble sort
		double dAux,vAux[dimsImage];
		int flag = 1;//indicates if any swapping has occurred
		int posAux;
		if(order<0)//descend order
		{
			while(flag == 1)
			{
				flag =0;
				for(int ii=0;ii<dimsImage-1;ii++)
				{
					if(d[ii]<d[ii+1])//swap
					{
						flag = 1;
						dAux = d[ii];d[ii] = d[ii+1]; d[ii+1] = dAux;//swap eigenvalues
						posAux = ii*dimsImage;
						for(int jj=0;jj<dimsImage;jj++)//swap eigenvectors
						{
							vAux[jj]=v[posAux];
							v[posAux] = v[posAux+dimsImage];
							v[posAux+dimsImage] = vAux[jj];
							posAux++;
						}
					}
				}
			}
		}else if(order>0)//ascendant order
		{
			while(flag == 1)
			{
				flag =0;
				for(int ii=0;ii<dimsImage-1;ii++)
				{
					if(d[ii]>d[ii+1])//swap
					{
						flag = 1;
						dAux = d[ii];d[ii] = d[ii+1]; d[ii+1] = dAux;//swap eigenvalues
						posAux = ii*dimsImage;
						for(int jj=0;jj<dimsImage;jj++)//swap eigenvectors
						{
							vAux[jj]=v[posAux];
							v[posAux] = v[posAux+dimsImage];
							v[posAux+dimsImage] = vAux[jj];
							posAux++;
						}
					}
				}
			}
		}

		__syncthreads();//to ensure coalescent memory access
		//copy back to global memory
		pos=tid;
		for(int ii=0;ii<dimsImage;ii++)
		{
			dCUDA[pos] = d[ii];//for coalescencent access efficiency
			pos+=numEllipsoids;
		}
		pos=tid;
		for(int ii=0;ii<dimsImage*dimsImage;ii++)
		{
			vCUDA[pos] = v[ii];
			pos+=numEllipsoids;
		}
	}
}

 //symmetrize (flip direction) if necessary
 __global__ void __launch_bounds__(MAX_THREADS_CUDA) applySymmetryToEigenvectorsKernel(double* vCUDA,int numEllipsoids,int symmetry)
 {
	 int tid=threadIdx.x + blockIdx.x * blockDim.x;
	 int vCUDAsize = numEllipsoids*dimsImage*dimsImage; 

	 if(tid<vCUDAsize)
	 {
		 //find out in which position we are within vCUDA we are in
		 int ellipsoidIdx = tid%numEllipsoids;
		 int vPos = (tid-ellipsoidIdx)/numEllipsoids;

		 //decide if it needs to be flipped or not
		 switch(vPos%dimsImage)
		 {
		 case 0://this is a v element that will multiply X value so we need to check if X needs to be flipped
			 if((symmetry & 0x01))
				 vCUDA[tid] *= -1.0;
			 break;
		 case 1://this is a v element that will multiply Y value so we need to check if Y needs to be flipped
			 if((symmetry & 0x02))
				 vCUDA[tid] *= -1.0;
			 break;
		 case 2://this is a v element that will multiply Z value so we need to check if Z needs to be flipped
			 if((symmetry & 0x04))
				 vCUDA[tid] *= -1.0;
			 break;
		 }		 
	 }
 }
//===============================end of eigenvalues kernel============================================================

//==============================beginning of interpolation kernel===============================================
//when you defined the properties of the texture textureImage you decide what to do with out of bounds values and if you prefer linear or nearest neighbor interpolation
__global__ void __launch_bounds__(MAX_THREADS_CUDA) interpolate3DBoxKernel(double* mCUDA,double* vCUDA,int ellipseIdx,int numEllipsoids,float* boxCUDA,int radiusBox_0, int radiusBox_1, int radiusBox_2,float* meanBoxCUDA,float* stdBoxCUDA)
{
	int boxSize=(1+2*radiusBox_0)*(1+2*radiusBox_1)*(1+2*radiusBox_2);
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ float m[dimsImage];
	__shared__ float v[dimsImage*dimsImage];//eigen vectors of the covariance matrix

	__shared__ float meanBox[MAX_THREADS_CUDA];//stores partial sums of the intensity to normalize the box
	__shared__ float stdBox[MAX_THREADS_CUDA];//stores partial sums of the intensity^2 to normalize the box

	//read from global memory to shared memory: unfortunately we do not have coalescent access
	if(threadIdx.x < dimsImage)
	{
		m[threadIdx.x]=mCUDA[threadIdx.x*numEllipsoids+ellipseIdx];
	}else if(threadIdx.x <dimsImage*(1+dimsImage))
	{
		v[threadIdx.x-dimsImage] = vCUDA[(threadIdx.x-dimsImage)*numEllipsoids+ellipseIdx];
	}

	meanBox[threadIdx.x]=0.0f;
	stdBox[threadIdx.x]=0.0f;
	__syncthreads();

	//calculate interpolation
	if(tid<boxSize)
	{
		float val,xi,yi,zi;
		int x,y,z,aux,aux2;

		aux=(1+2*radiusBox_0);
		x=tid%aux;
		aux2=(tid-x)/aux;
		aux=(1+2*radiusBox_1);
		y=aux2%aux;


		//if(dimsImage==3)
		//{
			z=(aux2-y)/aux;
			//center coordinates
			x-=radiusBox_0;
			y-=radiusBox_1;
			z-=radiusBox_2;
			xi=v[0]*x+v[1]*y+v[2]*z+m[0];
			yi=v[3]*x+v[4]*y+v[5]*z+m[1];
			zi=v[6]*x+v[7]*y+v[8]*z+m[2];
			val=tex3D(textureImage,xi+0.5f,yi+0.5f,zi+0.5f);//For CUDA, NN means just ceiling the coordinates, so we add 0.5 to really make it NN

		//}
		/*
		else if(dimsImage==2)
		{
			xi=v[0]*x+v[1]*y+m[0];
			yi=v[3]*x+v[4]*y+m[1];
			val=tex2D(textureImage,xi,yi); //I can not use tex2D with a 3D binded texture
		}
		*/


		boxCUDA[tid]=val;//coalescent access to save global memory

		//box statistics
		meanBox[threadIdx.x]=val;
		stdBox[threadIdx.x]=val*val;
	}

	__syncthreads();
	//add up the box statistics for each block
	int aux = blockDim.x/2;
	int aux2 = threadIdx.x;
	while (aux != 0) 
	{
		if (aux2 < aux)
		{
			stdBox[aux2] += stdBox[aux2 + aux];
			meanBox[aux2] += meanBox[aux2 + aux];
		}
		__syncthreads();
		aux /= 2;
	}

	if (aux2 == 0)
	{
		stdBoxCUDA[blockIdx.x] = stdBox[0];
		meanBoxCUDA[blockIdx.x] = meanBox[0];
	}
}
//==============================end of interpolation kernel===============================================

__global__ void __launch_bounds__(MAX_THREADS_CUDA) addBoxIntensityStatisticsKernel(float* meanBoxCUDA,float* stdBoxCUDA,float* meanFinalCUDA,float* stdFinalCUDA,int boxSize)
{
	__shared__ float meanFinal[sizeMeanStdBoxVector];
	__shared__ float stdFinal[sizeMeanStdBoxVector];

	//copy values
	meanFinal[threadIdx.x] = meanBoxCUDA[threadIdx.x];
	stdFinal[threadIdx.x] = stdBoxCUDA[threadIdx.x];

	__syncthreads();

	//add up the box statistics for each block
	int aux = blockDim.x/2;
	int aux2 = threadIdx.x;
	while (aux != 0) 
	{
		if (aux2 < aux)
		{
			stdFinal[aux2] += stdFinal[aux2 + aux];
			meanFinal[aux2] += meanFinal[aux2 + aux];
		}
		__syncthreads();
		aux /= 2;
	}

	if (aux2 == 0)
	{
		aux = boxSize-1;
		float auxS = meanFinal[0] / (float)boxSize;
		meanFinalCUDA[0] = auxS;
		auxS = stdFinal[0]/((float)aux)-auxS*auxS*((float)boxSize)/((float)aux);
		if(auxS < 1e-3)
			stdFinalCUDA[0] = 1.0;//if the Gausian is too small we might have nothing
		else
			stdFinalCUDA[0] = sqrt(auxS);
	}
}

//=========================beginning of box intensity statistics kernel=====================================
__global__ void __launch_bounds__(MAX_THREADS_CUDA) normalizeBoxKernel(float* boxCUDA,int boxSizeAux,float* meanFinalCUDA,float* stdFinalCUDA)
{
	__shared__ float mean;
	__shared__ float std;

	if(threadIdx.x==0)
		mean = meanFinalCUDA[0];
	if(threadIdx.x==1)
		std = stdFinalCUDA[0];

	__syncthreads();

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tid<boxSizeAux)
	{
		boxCUDA[tid]=(boxCUDA[tid]-mean)/std;
	}

}

//=========================end of box inetnsity statistics kernel =====================================

//=========================beginning of kernel to calculate cell idx for each voxel in a box================
__global__ void __launch_bounds__(MAX_THREADS_CUDA) calculateCellIdx3DKernel(double* dCUDA,int radiusBox_0, int radiusBox_1, int radiusBox_2,int ellipseIdx, int numEllipsoids,unsigned short int* boxCellIdxCUDA)
{
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ float d[dimsImage];//eigenvalues
	__shared__ int boxSize;
	
	//read from global memory to shared memory: unfortunately we do not have coalescent access
	if(threadIdx.x < dimsImage)
	{
		d[threadIdx.x] = dCUDA[threadIdx.x*numEllipsoids+ellipseIdx];
	}
	if(threadIdx.x == dimsImage)
		boxSize = (1+2*radiusBox_0)*(1+2*radiusBox_1)*(1+2*radiusBox_2);

	__syncthreads();


	//calculate interpolation
	if(tid<boxSize)
	{		
		float x,y,z;
		int aux,aux2;
		float rr;
		double xyz[dimsImage];
		long long int auxIdx;

		//calculate x,y,z coordinates for the point
		aux=(1+2*radiusBox_0);
		x=(float)(tid%aux);
		aux2=(tid-(int)x)/aux;
		aux=(1+2*radiusBox_1);
		y=(float)(aux2%aux);
		z=(float)((aux2-(int)y)/aux);
		//recenter and recover anysotropy from teh Gaussian
		x=sqrt(d[0])*(x-radiusBox_0);
		y=sqrt(d[1])*(y-radiusBox_1);
		z=sqrt(d[2])*(z-radiusBox_2);

		//calculate radius
		rr=sqrt((float)(x*x+y*y+z*z));

		//calculate normalized and centered (on a sphere) coordinates
		xyz[0] = x/rr;
		xyz[1] = y/rr;
		xyz[2] = z/rr;


		aux = 1;
		aux << numAngCells;//equivalent to (long long int)pow(2.0f,numAngCells) . Cuda 5.0 release compiled version does not retirn correct answer with pow function
		if(rr<1e-3)//otherwise it accesses out of bounds memory
			auxIdx = 1;
		else
			vec2pix_nest(aux, xyz, &auxIdx);

		//decide which cell belongs to
		if(rr>=r0CUDA[0]+(numRadialCells-1)*kSigmaCUDA[0])
			auxIdx = 0;//outside the largest ellipsoid
		else if(rr < r0CUDA[0])
			auxIdx = 1;//central ellipsoid
		else{
			aux = (int)floor((rr-r0CUDA[0])/kSigmaCUDA[0]);
			auxIdx = 2 + numCellsHEALPixCUDA[0]*aux + auxIdx;  
		}

		__syncthreads();//to guarantee coalescent access to memory
		//copy data
		boxCellIdxCUDA[tid] = auxIdx;		
	}
}

//=======================================================================================================
//each grop of threads within a block calculates the values of convolution in a whole column along the direction of the separable kernel. It does not provide coalescent memory access along Y and Z axis but we can use the shared memory to store all the column at once
__global__ void __launch_bounds__(MAX_THREADS_CUDA) separableConvolutionKernel(float* volCUDA,int dims_0,int dims_1,int dims_2,int kernelRadius,int dim)
{
	
	__shared__ int kernelSize;

	__shared__ float colConv[maxDiameterBox];
	__shared__ float colOrig[maxDiameterBox];//so we do not overwrite values while computing convolution

	__shared__ int dimsShared[3];//just we can write the code with for loops

	if(threadIdx.x == 0)
		dimsShared [threadIdx.x] = dims_0;
	else if(threadIdx.x == 1)
		dimsShared [threadIdx.x] = dims_1;
	else if(threadIdx.x == 2)
		dimsShared [threadIdx.x] = dims_2;
	else if(threadIdx.x == 3)
		kernelSize = 1+2*kernelRadius;
	__syncthreads();

	
	int iniPosXYZ [dimsImage];
	int offset = (dim+1)%dimsImage;
	iniPosXYZ[dim] = 0; //the initial pos is a plane of dim-th dimension = 0
	iniPosXYZ[offset] = blockIdx.x % dimsShared[offset];// generate a grid withthe other dimensions
	iniPosXYZ[(dim+2)%dimsImage] = (blockIdx.x -iniPosXYZ[offset])/ dimsShared[offset]; 

	int pos = 0;
	int dd = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		pos += dd * iniPosXYZ[ii];
		dd *= dimsShared[ii];
	}

	offset = 1;//how much do we have to skip to find next value along the separable kernel axis
	for(int ii=0;ii<dim;ii++)//for ii=0->offset=1->coalescent memory access
		offset *= dimsShared[ii];

	pos += offset*threadIdx.x;//add offset for each particular thread
	//reset values	
	colOrig[threadIdx.x] = volCUDA[pos];
	colConv[threadIdx.x] = 0;

	__syncthreads();

	//calculate value of each pixel
	offset = threadIdx.x - kernelRadius;
	float auxVal;
	for(int ii=0;ii<kernelSize;ii++)
	{
		if(offset<0) //make sure we are within bounds. TODO: allow different boundary conditions. Right no we are extending the last value.
			auxVal = colOrig[0];
		else if(offset>= dimsShared[dim])
			auxVal = colOrig[dimsShared[dim]-1];
		else
			auxVal = colOrig[offset];
		
		colConv[threadIdx.x] += auxVal * kernelSeparableConv[ii];
		
		offset++;
	}

	__syncthreads();

	//copy result back to global memory
	volCUDA[pos] = colConv[threadIdx.x];

}
//======================================================================================================

//========================kernel to calculate the average intensity in each sector======================
__global__ void __launch_bounds__(MAX_THREADS_CUDA) countIntensityPerCellKernel(float* fCellVecCUDA,int* nCellVecCUDA,float* boxCUDA,unsigned short int* boxCellIdxCUDA,int numCellsIdx,int boxSize)
{
	__shared__ float fCellVecCUDAshared[MAX_NUM_CELLS_IDX];
	__shared__ int nCellVecCUDAshared[MAX_NUM_CELLS_IDX];

	//reset values
	if(threadIdx.x<numCellsIdx)
	{
		fCellVecCUDAshared[threadIdx.x] = 0.0f;
		nCellVecCUDAshared[threadIdx.x] = 0;
	}

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid<boxSize)
	{
		unsigned short int nAux = boxCellIdxCUDA[tid];
		if(nAux>0)//we do not care about outside voxels label as zero cell index (we save a lot of atomic additions)
		{
			float fAux = boxCUDA[tid];
			atomicAdd(&(fCellVecCUDAshared[nAux]), fAux);
			atomicAdd(&(nCellVecCUDAshared[nAux]), 1);
		}
	}
	__syncthreads();

	//add to the global variable memory
	if(threadIdx.x<numCellsIdx)
	{
		if(nCellVecCUDAshared[threadIdx.x]>0)
		{
			atomicAdd(&(fCellVecCUDA[threadIdx.x]), fCellVecCUDAshared[threadIdx.x]);
			atomicAdd(&(nCellVecCUDA[threadIdx.x]), nCellVecCUDAshared[threadIdx.x]);
		}
	}
}

//====================end of kernel to calculate the average intensity in each sector======================

//========================================================================================================
void calculateSeparableConvolutionBoxInCUDA(float* volCUDA,const int* volRadiusDims,const double* d,cudaStream_t &stream)
{
	const int KsigmaDoG = 5;//kSigma*sigma defines the length of the Gaussian kernel
	int volDims[dimsImage];
	for(int ii=0;ii<dimsImage;ii++)
		volDims[ii] = 1+2*volRadiusDims[ii];

	if(maxDiameterBox>MAX_THREADS_CUDA)
	{
		cout<<"ERROR: at calculateSeparableConvolutionBoxInCUDA. Box cannot be larger than MAX_THREADS_CUDA for this particular implementation of convolution"<<endl;
		exit(3);
	}

	if(maxDiameterBox*maxDiameterBox>MAX_BLOCKS_CUDA)
	{
		cout<<"ERROR: at calculateSeparableConvolutionBoxInCUDA. Box cannot be larger than MAX_BLOCKS_CUDA for this particular implementation of convolution"<<endl;
		exit(3);
	}

	int volSize = volDims[0];
	for(int ii=1;ii<dimsImage;ii++)
		volSize *= volDims[ii];


	int kernelRadius=0;
	float sigmaDoG = 0.0f, diameter = 0.0f;
	float* kernelHOST = new float[1+2*maxRadiusBox];//this is the maximum size for the kernel
	float w = 0.0f;
	//calculate separable convolution for the first sigma
	for (int ii=0;ii<dimsImage;ii++)
	{
		//calculate kernel
		diameter=constDoG_0*sqrt(1.0/d[ii]);//usually when we plot Gaussians they are between 2-3 sigmas
		sigmaDoG=std::max(diameter/constDoG_1,1.0f);
		sigmaDoG=std::min(sigmaDoG,(floor(float(maxRadiusBox)/((float)KsigmaDoG*1.6f))));
		
		kernelRadius = (int)ceil(KsigmaDoG * sigmaDoG);
		w = 0.0f;
		for(int jj=0;jj<1+2*kernelRadius;jj++)
		{
			kernelHOST[jj] = exp(-0.5f*pow((jj-kernelRadius)/sigmaDoG,2));
			w += kernelHOST[jj];
		}
		for(int jj=0;jj<1+2*kernelRadius;jj++)//normalize
			kernelHOST[jj] /= w;
		//copy kernel to constant memory
		HANDLE_ERROR( cudaMemcpyToSymbol(kernelSeparableConv,kernelHOST,sizeof(float)*(1+2*kernelRadius)));
		//calculate convolution
		separableConvolutionKernel<<<volSize/volDims[ii],volDims[ii],0,stream>>>(volCUDA,volDims[0],volDims[1],volDims[2],kernelRadius,ii);HANDLE_ERROR_KERNEL;
	}
	//calculate separable convolution for the second sigma
	for (int ii=0;ii<dimsImage;ii++)
	{
		//calculate kernel
		diameter=constDoG_0*sqrt(1.0/d[ii]);//usually when we plot Gaussians they are between 2-3 sigmas
		sigmaDoG=std::max(diameter/constDoG_1,1.0f);
		sigmaDoG=std::min(sigmaDoG,(floor(float(maxRadiusBox)/((float)KsigmaDoG*1.6f))));
		sigmaDoG *=1.6f;//for the DoG
		
		kernelRadius = (int)ceil(KsigmaDoG * sigmaDoG);
		w = 0.0f;
		for(int jj=0;jj<1+2*kernelRadius;jj++)
		{
			kernelHOST[jj] = exp(-0.5f*pow((jj-kernelRadius)/sigmaDoG,2));
			w += kernelHOST[jj];
		}
		for(int jj=0;jj<1+2*kernelRadius;jj++)//normalize
			kernelHOST[jj] /= (-w);//negative so we achieve the DoG effect
		//copy kernel to constant memory
		HANDLE_ERROR( cudaMemcpyToSymbol(kernelSeparableConv,kernelHOST,sizeof(float)*(1+2*kernelRadius)));
		//calculate convolution
		separableConvolutionKernel<<<volSize/volDims[ii],volDims[ii],0,stream>>>(volCUDA,volDims[0],volDims[1],volDims[2],kernelRadius,ii);HANDLE_ERROR_KERNEL;
	}

	delete[] kernelHOST;
}

//---------------------------------------------------------------------------

basicEllipticalHaarFeatureVector** calculateEllipticalHaarFeatures(const double *m,const double *W,int numEllipsoids,const imageType *im,const long long int *dims,int devCUDA,int symmetry)
{
	basicEllipticalHaarFeatureVector **f=NULL;//vector containing the final value fo features
	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	if(sizeMeanStdBoxVector>MAX_THREADS_CUDA)
	{
		cout<<"ERROR: sizeMeanStdBoxVecotr cannot be bigger than MAX_THREADS_CUDA"<<endl;
		return f;
	}
	if(MAX_NUM_CELLS_IDX<numCellsIdx)
	{
		cout<<"ERROR: MAX_NUM_CELLS_IDX < numCellsIdx. Chang ethe maximum size so we can allocate share memory for reduction purposes"<<endl;
		return f;
	}

	//---------------allocate memory in GPU and transfer the data----------------------------------------------
	//cout<<"Allocating memory in devCUDA="<<devCUDA<<endl;

	//initialize constant memory
	HANDLE_ERROR( cudaMemcpyToSymbol(r0CUDA,&r0,sizeof(double)));
	HANDLE_ERROR( cudaMemcpyToSymbol(kSigmaCUDA,&kSigma,sizeof(double)));
	HANDLE_ERROR( cudaMemcpyToSymbol(numCellsHEALPixCUDA,&numCellsHEALPix,sizeof(int)));

	int x2pixHOST[128];//HEALPIX constants
	int y2pixHOST[128];
	mk_xy2pix(x2pixHOST,y2pixHOST);
	HANDLE_ERROR( cudaMemcpyToSymbol(x2pix,x2pixHOST,128*sizeof(int)));
	HANDLE_ERROR( cudaMemcpyToSymbol(y2pix,y2pixHOST,128*sizeof(int)));

	//imageType *imCUDA = NULL;
	cudaArray* imCUDA = NULL;

	double *mCUDA = NULL;//mean
	double *wCUDA = NULL;//covariance
	
	double *dCUDA = NULL; //eigenvalues
	double *vCUDA = NULL; //eigenvectors

	//boxes are not declared as cudaArray (or binded to texture) because we cannot write on cudaArrays
	float *boxCUDA = NULL;//holds the interpolated box for each Gaussian
	float* stdBoxCUDA = NULL;//partial sums of intensity^2 values to normalize box
	float* meanBoxCUDA = NULL;//partial sums of intensity values to normalize box
	float* meanFinalCUDA = NULL;//stores final value for mean
	float* stdFinalCUDA = NULL;//stores final value for mean
	unsigned short int* boxCellIdxCUDA = NULL;//contains the cell id for each voxel, so we can calculate features

	long long int imSize = dims[0];
	int maxBoxSize = maxDiameterBox;
	for(int ii=1;ii<dimsImage;ii++)
	{
		imSize *= dims[ii];
		maxBoxSize *= maxDiameterBox;
	}
	//int maxDiameterBox=2*maxRadiusBox+1;

	//allocate cuda array for image and for box(texture memory)
	// allocate CudaArray
	cudaChannelFormatDesc channelDescImage = cudaCreateChannelDesc<imageType>();
	
	
	//create a cudaExtent structure, storing the dimensions of the 3D texture
	cudaExtent imageSize = make_cudaExtent(dims[0], dims[1], dims[2]);
	

	if(dimsImage==3)
	{
		HANDLE_ERROR(cudaMalloc3DArray(&imCUDA, &channelDescImage, imageSize));
	
		/*create a cudaMemcpy3DParms structures--this is a structure that tells cuda how to copy data using the cudaMemcpy3D function. Basically, this prevents having to pass a bunch of parameters in favor of a single complex structure.*/
		cudaMemcpy3DParms copyParmsImage={0};
		copyParmsImage.srcPtr = make_cudaPitchedPtr((void*)im, dims[0]*sizeof(imageType), dims[0], dims[1]); 
		copyParmsImage.dstArray = imCUDA;
		copyParmsImage.extent = imageSize;
		copyParmsImage.kind = cudaMemcpyHostToDevice;
		HANDLE_ERROR( cudaMemcpy3D(&copyParmsImage));
		
	}else{//TODO: do it for 2D;
		cout<<"ERROR: code is not ready for dimsImage="<<dimsImage<<endl;
		return f;
	}
	
	//set the parameters for the global texture variable
	textureImage.normalized = false; //coordinates are not between [0,1]^dimsImage but between image size boundaries
	textureImage.filterMode = cudaFilterModePoint;//nearest neighbor interpolation. Use cudaFilterModeLinear for linear interpolation
	textureImage.addressMode[0] = cudaAddressModeClamp;//How out of bounds requests are handled. For non-normalized mode only clamp is supported. In clamp addressing mode x is replaced by 0 if x<0 and N-1 if x>=N;
	textureImage.addressMode[1] = cudaAddressModeClamp;
	textureImage.addressMode[2] = cudaAddressModeClamp;

	
	//bind the texture to the array
	HANDLE_ERROR( cudaBindTextureToArray(textureImage, imCUDA, channelDescImage) );


	//allocate memory to hold the image and boxes (we just allocate the max allowed since it is small enough)	
	//HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSize*sizeof(imageType) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(boxCUDA), maxBoxSize*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(boxCellIdxCUDA), maxBoxSize*sizeof(unsigned short int) ) );
	
	//int sizeMeanStdBoxVector = (int)ceil(((float)(maxDiameterBox*maxDiameterBox*maxDiameterBox)/((float)MAX_THREADS_CUDA));
	//we need it ot be a power of 2
	//sizeMeanStdBoxVector = (int)pow(2.0f,(int)ceil(log2((float)sizeMeanStdBoxVector)));
	
	HANDLE_ERROR( cudaMalloc( (void**)&(meanBoxCUDA), sizeMeanStdBoxVector*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(stdBoxCUDA), sizeMeanStdBoxVector*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(meanFinalCUDA), sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(stdFinalCUDA), sizeof(float) ) );
	

	//allocate memory for Gaussian centroids and covariance
	HANDLE_ERROR( cudaMalloc( (void**)&(mCUDA), numEllipsoids*dimsImage*sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(wCUDA), numEllipsoids*dimsImage*((dimsImage+1)/2)*sizeof(double) ) );

	//copy to GPU
	//HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSize*sizeof(imageType) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( mCUDA, m, numEllipsoids*dimsImage*sizeof(double) , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( wCUDA, W, numEllipsoids*dimsImage*((dimsImage+1)/2)*sizeof(double) , cudaMemcpyHostToDevice ) );

	//allocate memory for eigenvalues and eigenvectors
	HANDLE_ERROR( cudaMalloc( (void**)&(dCUDA), numEllipsoids*dimsImage*sizeof(double) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(vCUDA), numEllipsoids*dimsImage*dimsImage*sizeof(double) ) );

	//generate strems to parallelize kernel and data transfer
	cudaStream_t stream0, stream1;
	HANDLE_ERROR( cudaStreamCreate( &stream0));
	HANDLE_ERROR( cudaStreamCreate( &stream1));

	float* fCellVecHOST = NULL;//stores the sum of intensities for each cell
	int* nCellVecHOST = NULL;//stores the weights so we can calculate average
	float* fCellVecHOSTDoG = NULL;//stores the sum of intensities for each cell
	int* nCellVecHOSTDoG = NULL;//stores the weights so we can calculate average

	float* fCellVecCUDA = NULL;//stores the sum of intensities for each cell
	int* nCellVecCUDA = NULL;//stores the weights so we can calculate average
	float* fCellVecCUDADoG = NULL;//stores the sum of intensities for each cell
	int* nCellVecCUDADoG = NULL;//stores the weights so we can calculate average
	//allocate as pinned memory so transfers are faster and we can use streams to launch kernels in parallel with data transfer (Chapter 10 of CUDA By example book)
	HANDLE_ERROR( cudaHostAlloc( (void**) &fCellVecHOST, numCellsIdx*sizeof(float),cudaHostAllocDefault));
	HANDLE_ERROR( cudaHostAlloc( (void**) &nCellVecHOST, numCellsIdx*sizeof(int),cudaHostAllocDefault));
	HANDLE_ERROR( cudaMalloc( (void**)&(fCellVecCUDA), numCellsIdx*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(nCellVecCUDA), numCellsIdx*sizeof(int)) );
	HANDLE_ERROR( cudaHostAlloc( (void**) &fCellVecHOSTDoG, numCellsIdx*sizeof(float),cudaHostAllocDefault));
	HANDLE_ERROR( cudaHostAlloc( (void**) &nCellVecHOSTDoG, numCellsIdx*sizeof(int),cudaHostAllocDefault));
	HANDLE_ERROR( cudaMalloc( (void**)&(fCellVecCUDADoG), numCellsIdx*sizeof(float) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(nCellVecCUDADoG), numCellsIdx*sizeof(int)) );

	//----------------calculate eigenvalues and eigenvectors for each Gaussian---------------
	int numThreads=std::min(MAX_THREADS_CUDA,numEllipsoids);
	int numBlocks=std::min(MAX_BLOCKS_CUDA,(numEllipsoids+numThreads-1)/numThreads);
	computeEigenDecompositionKernel<<<numBlocks,numThreads>>>(wCUDA,numEllipsoids,dCUDA,vCUDA,1);HANDLE_ERROR_KERNEL;//egigenvalues organized in ascend order
	//copy info back so we know dimensions for each box
	double *dHOST = new double[numEllipsoids*dimsImage];
	double *vHOST = new double[numEllipsoids*dimsImage*dimsImage];

	//apply symmetry if requested by user to artifically extend training data
	if(symmetry>7)
	{
		cout<<"ERROR: symmetry input cannot be higher than 7"<<endl;
		return f;
	}else if(symmetry>0){//apply symmetry before copying eigenvalues to host
		numThreads=std::min(MAX_THREADS_CUDA,numEllipsoids*dimsImage*dimsImage);
		numBlocks=std::min(MAX_BLOCKS_CUDA,(numEllipsoids*dimsImage*dimsImage+numThreads-1)/numThreads);
		applySymmetryToEigenvectorsKernel<<<numBlocks,numThreads>>>(vCUDA,numEllipsoids,symmetry);HANDLE_ERROR_KERNEL;//egigenvalues organized in ascend order
	}

	HANDLE_ERROR(cudaMemcpy(dHOST,dCUDA,sizeof(double)*numEllipsoids*dimsImage,cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(vHOST,vCUDA,sizeof(double)*numEllipsoids*dimsImage*dimsImage,cudaMemcpyDeviceToHost));

	//----------main for loop calculating features for each ellipsoid---------------------
	//cout<<"Starting main loop to calculate features for "<<numEllipsoids<<" boxes"<<endl;
	int radiusBox[dimsImage];
	int pos = 0;
	const double scaleSigma = r0+numRadialCells*kSigma;
	//allocate main memory for output
	f = new basicEllipticalHaarFeatureVector*[numEllipsoids];
	for(int ii=0;ii<numEllipsoids;ii++)
		f[ii] = new basicEllipticalHaarFeatureVector;

	for(int ii=0;ii<numEllipsoids;ii++)
	{
		
		//reset box statistics
		HANDLE_ERROR(cudaMemset(meanBoxCUDA,0,sizeof(float)*sizeMeanStdBoxVector));
		HANDLE_ERROR(cudaMemset(stdBoxCUDA,0,sizeof(float)*sizeMeanStdBoxVector));

		//reset cell idx and values before streams
		HANDLE_ERROR(cudaMemset(fCellVecCUDA,0,numCellsIdx*sizeof(float)));
		HANDLE_ERROR(cudaMemset(nCellVecCUDA,0,numCellsIdx*sizeof(int)));
		HANDLE_ERROR(cudaMemset(fCellVecCUDADoG,0,numCellsIdx*sizeof(float)));
		HANDLE_ERROR(cudaMemset(nCellVecCUDADoG,0,numCellsIdx*sizeof(int)));

		

		//calculate rotated box (interpolation)
		pos = ii;
	    int boxSizeAux = 1;
		for(int jj=0;jj<dimsImage;jj++)
		{
			radiusBox[jj] = (int)ceil(scaleSigma*sqrt(1.0/dHOST[pos]));
			radiusBox[jj] = std::max(radiusBox[jj],minRadiusBox);
			radiusBox[jj] = std::min(radiusBox[jj],maxRadiusBox);
			pos+=numEllipsoids;
			boxSizeAux *= (1+2*radiusBox[jj]);
		}
		numThreads=MAX_THREADS_CUDA; //we do reduction (i.e. we need a power of 2) and most of the time boxSizeAux>MAX_THREADS_CUDA, so we just set it to MAX_THREADS_CUDA
		numBlocks=std::min(MAX_BLOCKS_CUDA,(boxSizeAux+numThreads-1)/numThreads);
		

		if(dimsImage==3)
		{
			interpolate3DBoxKernel<<<numBlocks,numThreads>>>(mCUDA,vCUDA,ii,numEllipsoids,boxCUDA,radiusBox[0],radiusBox[1],radiusBox[2],meanBoxCUDA,stdBoxCUDA);
			HANDLE_ERROR_KERNEL;
		}
		else if(dimsImage==2)
		{
			//TODO: I cannot define a 2D and 3D texture atthe same time->I need to write separate codes or thing how to do it
			//interpolate2DBoxKernel<<<numBlocks,numThreads>>>(mCUDA,vCUDA,ii,numEllipsoids,boxCUDA,radiusBox[0],radiusBox[1],meanBoxCUDA,stdBoxCUDA);
			//HANDLE_ERROR_KERNEL;
			cout<<"ERROR: code is not ready for 2D interpolation"<<endl;
			for(int ii=0;ii<numEllipsoids;ii++) delete f[ii];
			delete[] f;
			f = NULL;
			return f;
		}

		//------------------------------------------------------------------------
		//normalize the box using the partial sums calculated during interpolation
		addBoxIntensityStatisticsKernel<<<1,sizeMeanStdBoxVector>>>(meanBoxCUDA,stdBoxCUDA,meanFinalCUDA,stdFinalCUDA,boxSizeAux);HANDLE_ERROR_KERNEL;
		normalizeBoxKernel<<<numBlocks,numThreads>>>(boxCUDA,boxSizeAux,meanFinalCUDA,stdFinalCUDA);HANDLE_ERROR_KERNEL;
		

		//----------------------------------------------------
		//calculate basic features values (sectors) for the box
		//figure out the cell for each voxel in the box
		calculateCellIdx3DKernel<<<numBlocks,numThreads>>>(dCUDA,radiusBox[0], radiusBox[1], radiusBox[2],ii, numEllipsoids,boxCellIdxCUDA);HANDLE_ERROR_KERNEL;
		
		
		//make sure GPU finishes before we launch two different streams
		HANDLE_ERROR(cudaDeviceSynchronize());	
		
		
		//-------------------debug-------------------------------
		/*
		cout<<"DEBUGGING: saving box to E:\\temp\\box.bin with size "<<(1+2*radiusBox[0])<<","<<(1+2*radiusBox[1])<<","<<(1+2*radiusBox[2])<<endl;
		float* boxHOST = new float[boxSizeAux];
		HANDLE_ERROR( cudaMemcpy( boxHOST, boxCUDA, boxSizeAux*sizeof(float) , cudaMemcpyDeviceToHost) );
		FILE* fid=fopen("E:/temp/box.bin","wb");
		fwrite(boxHOST,sizeof(float),boxSizeAux,fid);
		fclose(fid);
		delete[] boxHOST;

		unsigned short int* boxCellIdxHOST = new unsigned short int[boxSizeAux];
		HANDLE_ERROR( cudaMemcpy( boxCellIdxHOST, boxCellIdxCUDA, boxSizeAux*sizeof(unsigned short int) , cudaMemcpyDeviceToHost) );
		fid=fopen("E:/temp/boxCellIdx.bin","wb");
		fwrite(boxCellIdxHOST,sizeof(unsigned short int),boxSizeAux,fid);
		fclose(fid);
		delete[] boxCellIdxHOST;
		*/
		//------------------------------------------------------		
		

		//calculate values of f for each cell 
		countIntensityPerCellKernel<<<numBlocks,numThreads,0,stream0>>>(fCellVecCUDA,nCellVecCUDA,boxCUDA,boxCellIdxCUDA,numCellsIdx,boxSizeAux);
		HANDLE_ERROR( cudaMemcpyAsync( fCellVecHOST, fCellVecCUDA, numCellsIdx*sizeof(float) , cudaMemcpyDeviceToHost, stream0 ) );
		HANDLE_ERROR( cudaMemcpyAsync( nCellVecHOST, nCellVecCUDA, numCellsIdx*sizeof(int) , cudaMemcpyDeviceToHost, stream0 ) );

		//calculate DoG of the box
		double dAux[dimsImage];
		for(int jj=0;jj<dimsImage;jj++)
			dAux[jj] = dHOST[ii + numEllipsoids *jj];//select eigenvalues
		calculateSeparableConvolutionBoxInCUDA(boxCUDA,radiusBox,dAux,stream1);				
		countIntensityPerCellKernel<<<numBlocks,numThreads,0,stream1>>>(fCellVecCUDADoG,nCellVecCUDADoG,boxCUDA,boxCellIdxCUDA,numCellsIdx,boxSizeAux);

		//calculate values of f for each cell 
		HANDLE_ERROR( cudaMemcpyAsync( fCellVecHOSTDoG, fCellVecCUDADoG, numCellsIdx*sizeof(float) , cudaMemcpyDeviceToHost, stream1 ) );
		HANDLE_ERROR( cudaMemcpyAsync( nCellVecHOSTDoG, nCellVecCUDADoG, numCellsIdx*sizeof(int) , cudaMemcpyDeviceToHost, stream1 ) );

		
		HANDLE_ERROR( cudaStreamSynchronize( stream0));
		HANDLE_ERROR( cudaStreamSynchronize( stream1));
		
		
		//----------------------debug-------------------------
		/*
		float* boxHOST = new float[boxSizeAux];
		HANDLE_ERROR( cudaMemcpy( boxHOST, boxCUDA, boxSizeAux*sizeof(float) , cudaMemcpyDeviceToHost) );
		cout<<"DEBUGGING: saving box to E:\\temp\\boxDoG.bin with size "<<(1+2*radiusBox[0])<<","<<(1+2*radiusBox[1])<<","<<(1+2*radiusBox[2])<<endl;
		fid=fopen("E:/temp/boxDoG.bin","wb");
		fwrite(boxHOST,sizeof(float),boxSizeAux,fid);
		fclose(fid);
		delete [] boxHOST;
		
		cout<<"DEBUGGING: CUDA code elliptical features"<<endl;
		for(int hh = 0; hh<numCellsIdx; hh++)
		{
			cout<<fCellVecHOST[hh]<<" "<<nCellVecHOST[hh]<<" "<<fCellVecHOSTDoG[hh]<<" "<<nCellVecHOSTDoG[hh]<<endl;
		}
		exit (2);
		*/
		//-----------------------------------------------------
		


		//calculate excentricity 
		int count = 0;
		for(int jj=0;jj<dimsImage;jj++)
			for(int kk=jj+1;kk<dimsImage;kk++)
			{
				if(dAux[kk]<1e-10)
					f[ii]->excentricity[count] = 0.0f;
				else
					f[ii]->excentricity[count] = dAux[jj]/dAux[kk];

				count++;
			}
		//----------------------------------------------------
		//calculate all the cells and rings
		f[ii]->ringAvgIntensity[0] = fCellVecHOST[1] / (float)(nCellVecHOST[1]);//central ring
		f[ii]->ringAvgIntensityDoG[0] = fCellVecHOSTDoG[1] / (float)(nCellVecHOSTDoG[1]);//central ring
		count = 0;		
		int count2 = count + 2;
		for(int jj=0;jj<numRadialCells-1;jj++)
		{
			int Nring = 0, NringDoG = 0;
			float Wring = 0.0f, WringDoG = 0.0f;
			for(int kk=0;kk<numCellsHEALPix;kk++)
			{
				Nring += nCellVecHOST[count2];
				Wring += fCellVecHOST[count2];
				
				if( nCellVecHOST[count2] == 0)//sometimes ellipsois are so flat in Z that some cells do not get any pixel
					f[ii]->cellAvgIntensity[count] = 0;
				else
					f[ii]->cellAvgIntensity[count] = fCellVecHOST[count2] / (float)(nCellVecHOST[count2]);

				NringDoG += nCellVecHOSTDoG[count2];
				WringDoG += fCellVecHOSTDoG[count2];
				if( nCellVecHOSTDoG[count2] == 0)//sometimes ellipsois are so flat in Z that some cells do not get any pixel
					f[ii]->cellAvgIntensityDoG[count] = 0;
				else
					f[ii]->cellAvgIntensityDoG[count] = fCellVecHOSTDoG[count2] / (float)(nCellVecHOSTDoG[count2]);

				count++;
				count2++;
			}
			if( Nring == 0 )
				f[ii]->ringAvgIntensity[jj+1] = 0;
			else
				f[ii]->ringAvgIntensity[jj+1] = Wring/(float)Nring;

			if( NringDoG == 0 )
				f[ii]->ringAvgIntensityDoG[jj+1] = 0;
			else
				f[ii]->ringAvgIntensityDoG[jj+1] = WringDoG/(float)NringDoG;
		}
		//----------------------------------------------------
		//expand the set of features by combining pairs
		//TODO
	}

	//unbind textures
	cudaUnbindTexture(textureImage);

	//deallocate memory
	HANDLE_ERROR( cudaFree( mCUDA ) );
	HANDLE_ERROR( cudaFree( wCUDA ) );
	HANDLE_ERROR( cudaFree( dCUDA ) );
	HANDLE_ERROR( cudaFree( vCUDA ) );
	HANDLE_ERROR( cudaFree( meanBoxCUDA ) );
	HANDLE_ERROR( cudaFree( stdBoxCUDA ) );	
	HANDLE_ERROR( cudaFree( meanFinalCUDA ) );
	HANDLE_ERROR( cudaFree( stdFinalCUDA ) );
	HANDLE_ERROR( cudaFree( boxCUDA ) );
	HANDLE_ERROR( cudaFree( boxCellIdxCUDA ) );
	//HANDLE_ERROR( cudaFree( imCUDA ) );
	HANDLE_ERROR( cudaFreeArray( imCUDA ) );
	
	HANDLE_ERROR( cudaStreamDestroy( stream0));
	HANDLE_ERROR( cudaStreamDestroy( stream1));

	delete[] dHOST;
	delete[] vHOST;
	HANDLE_ERROR( cudaFreeHost( fCellVecHOST ));
	HANDLE_ERROR( cudaFreeHost( nCellVecHOST ));
	HANDLE_ERROR( cudaFree( fCellVecCUDA));
	HANDLE_ERROR( cudaFree( nCellVecCUDA));
	HANDLE_ERROR( cudaFreeHost( fCellVecHOSTDoG ));
	HANDLE_ERROR( cudaFreeHost( nCellVecHOSTDoG ));
	HANDLE_ERROR( cudaFree( fCellVecCUDADoG));
	HANDLE_ERROR( cudaFree( nCellVecCUDADoG));


	return f;
}



/*
	//--------------------------------------debug: cuda3Dtexture------------------------
		cout<<"DEBUGGING: 3D texture access. Writing copied file out"<<endl;
		imageType* imAuxCUDA, *imAuxHOST;
		imAuxHOST = new imageType[imSize];
		HANDLE_ERROR( cudaMalloc( (void**)&(imAuxCUDA),imSize*sizeof(imageType) ) );

		int numThreadsAux=std::min(MAX_THREADS_CUDA,(int)imSize);
		int numBlocksAux=std::min(MAX_BLOCKS_CUDA,((int)imSize+numThreadsAux-1)/numThreadsAux);

		debuggingCopy3DTextureKernel<<<numBlocksAux,numThreadsAux>>>(imAuxCUDA,imSize,dims[0],dims[1],dims[2]);HANDLE_ERROR_KERNEL;
		HANDLE_ERROR( cudaMemcpy( imAuxHOST, imAuxCUDA, imSize*sizeof(imageType) , cudaMemcpyDeviceToHost ) );

		FILE* fim=fopen("E:/temp/imCUDAcopy.bin","wb");
		fwrite(imAuxHOST,sizeof(imageType),imSize,fim);
		fclose(fim);

		fim=fopen("E:/temp/imHOSTcopy.bin","wb");
		fwrite(im,sizeof(imageType),imSize,fim);
		fclose(fim);
		//-------------------------------------------------------------------

		*/

/*
//=======================================debug kernel========================================================
__global__ void __launch_bounds__(MAX_THREADS_CUDA) debuggingCopy3DTextureKernel (imageType* imAuxCUDA,long long int imSize,int dims_0,int dims_1,int dims_2)
{
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	float x,y,z;
	int aux,aux2;
		
	while(tid<imSize)
	{
		

		//calculate x,y,z coordinates for the point
		aux=dims_0;
		x=(float)(tid%aux);
		aux2=(tid-(int)x)/aux;
		aux=dims_1;
		y=(float)(aux2%aux);
		z=(float)((aux2-(int)y)/aux);

		imAuxCUDA[tid] = tex3D(textureImage,x,y,z);

		if(tid==0)
		{
			printf("Value at (%f,%f,%f) from debuggin kernel is %d\n",x,y,z,int(imAuxCUDA[tid]));
		}

		tid+= blockDim.x * gridDim.x;
	}
}

//=========================================================================================
*/