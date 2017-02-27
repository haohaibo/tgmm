/*
 * responsibilities.cpp
 *
 *  Created on: May 12, 2011
 *      Author: amatf
 */

#include "responsibilities.h"
#include <fstream>


const float colormap::hsv18[54]={1.0000    ,     0     ,    0  ,  1.0000  ,  0.3333   ,      0  ,  1.0000   , 0.6667    ,     0  ,  1.0000 ,   1.0000      ,   0,
		0.6667  ,  1.0000   ,      0  ,  0.3333 ,   1.0000     ,    0  ,       0  ,  1.0000    ,     0    ,     0  ,  1.0000 ,   0.3333,
		 0  ,  1.0000  ,  0.6667   ,      0   , 1.0000  ,  1.0000     ,    0 ,   0.6667 ,   1.0000   ,      0  ,  0.3333  ,  1.0000,
		 0    ,     0  ,  1.0000  ,  0.3333      ,   0 ,   1.0000   , 0.6667     ,    0  ,  1.0000 ,   1.0000   ,      0  ,  1.0000,
		 1.0000  ,       0   , 0.6667 ,   1.0000     ,    0  ,  0.3333};//18x3=54;

responsibilities::responsibilities(unsigned long long int N_,unsigned long long int K_, long long int nzmax)
{
	N=N_;
	K=K_;
	R_nk=cs_spalloc(N,K,nzmax);

	if(R_nk==NULL)
	{
		cout<<"ERROR: at responsibilities::responsibilities(unsigned long long int N_,unsigned long long int K_) allocating sparse matrix "<<endl;
	}
}


//=========================================================
responsibilities::~responsibilities()
{
	R_nk=cs_spfree(R_nk);//returns NULL. Already protected in case R-nk is null
}


//==========================================================
void responsibilities::writeOutMatlabFormat(string fileOut)
{
	cout<<"DEBUGGING: responsibilities::writeOutMatlabFormat"<<endl;
	ofstream out(fileOut.c_str());

	//out<<"rr=["<<endl;

	for(int ii=0;ii<R_nk->nz;ii++)
	{
		//out<<R_nk->i[ii]<<" "<<R_nk->p[ii]<<" "<<R_nk->x[ii]<<";"<<endl;
		out<<R_nk->i[ii]<<" "<<R_nk->p[ii]<<" "<<R_nk->x[ii]<<endl;
	}

	//out<<"];"<<endl;
	out.close();
}

//===========================================================
void responsibilities::writeOutSegmentationMaskAndImageBlend(mylib::Array *img,string fileout,const vector<colormap> &colMap)
{
	const double alpha=0.65;//blend weight
	const double alpha_c=1.0-alpha;
	const double uint8=255;
	const double minR_nkValue=0.1;//minimum probability to be accepted as a pixel

	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	mylib::Array *blend=mylib::Make_Array(mylib::RGB_KIND,mylib::UINT8_TYPE,img->ndims,img->dims);

	mylib::Array_Bundle blen_R = *blend;  // Red channel
	Get_Array_Plane(&blen_R,1);
	mylib::Array_Bundle blen_G = *blend;  // Green channel
	Get_Array_Plane(&blen_G,2);
	mylib::Array_Bundle blen_B = *blend;  // Blue channel
	Get_Array_Plane(&blen_B,3);


	//copy image to RGB blend
	mylib::uint8 *blend_RData=(mylib::uint8*)(blen_R.data);
	mylib::uint8 *blend_GData=(mylib::uint8*)(blen_G.data);
	mylib::uint8 *blend_BData=(mylib::uint8*)(blen_B.data);
	mylib::uint8 auxI;
	for(mylib::Size_Type ii=0;ii<img->size;ii++)
	{
		auxI=(mylib::uint8)(alpha*uint8*imgData[ii]);
		blend_RData[ii]=auxI;
		blend_GData[ii]=auxI;
		blend_BData[ii]=auxI;
	}


	int nOld=-1;
	unsigned int maxPos;
	vector< pair<int,double> > vecKr;
	vecKr.reserve(maxGaussiansPerVoxel);
	for(int ii=0;ii<R_nk->nz;ii++)//we assume R_nk is ordered in n
	{
		if(R_nk->i[ii]==nOld)//still accumulating values for the same n
		{
			vecKr.push_back(make_pair(R_nk->p[ii],R_nk->x[ii]));
		}else{

			if(!vecKr.empty())//create the blend for current pixel
			{
				maxPos=0;
				for(unsigned int kk=1;kk<vecKr.size();kk++)
				{
					if(vecKr[maxPos].second<vecKr[kk].second) maxPos=kk;
				}
				if(vecKr[maxPos].second>minR_nkValue)
				{
					if(vecKr[maxPos].first>=(int)colMap.size())
					{
						cout<<"ERROR: at writeOutSegmentationMaskAndImageBlend. Colormap is not long enough for all the clusters"<<endl;
						exit(2);
					}
					blend_RData[nOld]+=(mylib::uint8)(uint8*alpha_c*colMap[vecKr[maxPos].first].r);
					blend_GData[nOld]+=(mylib::uint8)(uint8*alpha_c*colMap[vecKr[maxPos].first].g);
					blend_BData[nOld]+=(mylib::uint8)(uint8*alpha_c*colMap[vecKr[maxPos].first].b);
				}
			}

			//start new sequence for new n
			nOld=R_nk->i[ii];
			vecKr.clear();
			vecKr.push_back(make_pair(R_nk->p[ii],R_nk->x[ii]));
		}
	}


	if(mylib::Write_Image((char*)fileout.c_str(), blend, mylib::DONT_PRESS))//write file with no compressions
	{
		cout<<"ERROR: at writeOutSegmentationMaskAndImageBlend file was not written properly"<<endl;
		exit(2);
	}
	mylib::Free_Array(blend);
};

//============================================================================================

void writeOutSegmentationMaskAndImageBlend(mylib::Array *img,string fileout,const vector<colormap> &colMap,long long int *nPos,int *kAssignment,int *colorVec,long long int query_nb)
{
	const double alpha=0.65;//blend weight
	const double alpha_c=1.0-alpha;
	const double uint8=255.0;

	mylib::float32 *imgData=(mylib::float32*)(img->data);
	if(img->type!=8)
	{
		cout<<"ERROR: code is only ready for FLOAT32 images"<<endl;
		exit(10);
	}

	mylib::Array *blend=mylib::Make_Array(mylib::RGB_KIND,mylib::UINT8_TYPE,img->ndims,img->dims);

	mylib::Array_Bundle blen_R = *blend;  // Red channel
	Get_Array_Plane(&blen_R,1);
	mylib::Array_Bundle blen_G = *blend;  // Green channel
	Get_Array_Plane(&blen_G,2);
	mylib::Array_Bundle blen_B = *blend;  // Blue channel
	Get_Array_Plane(&blen_B,3);


	//copy image to RGB blend
	mylib::uint8 *blend_RData=(mylib::uint8*)(blen_R.data);
	mylib::uint8 *blend_GData=(mylib::uint8*)(blen_G.data);
	mylib::uint8 *blend_BData=(mylib::uint8*)(blen_B.data);
	mylib::uint8 auxI;
	for(mylib::Size_Type ii=0;ii<img->size;ii++)
	{
		auxI=(mylib::uint8)(alpha*uint8*imgData[ii]);
		blend_RData[ii]=auxI;
		blend_GData[ii]=auxI;
		blend_BData[ii]=auxI;
	}

	long long int n;
	int colPos;
	for(long long int ii=0;ii<query_nb;ii++)
	{
		if(kAssignment[ii]<0) continue;
		n=nPos[ii];
		colPos=colorVec[kAssignment[ii]];
		blend_RData[n]+=(mylib::uint8)(uint8*alpha_c*colMap[colPos].r);
		blend_GData[n]+=(mylib::uint8)(uint8*alpha_c*colMap[colPos].g);
		blend_BData[n]+=(mylib::uint8)(uint8*alpha_c*colMap[colPos].b);
	}

	if(mylib::Write_Image((char*)fileout.c_str(), blend, mylib::DONT_PRESS))//write file with no compressions
	{
		cout<<"ERROR: at writeOutSegmentationMaskAndImageBlend file was not written properly"<<endl;
		exit(2);
	}
	mylib::Free_Array(blend);
}
