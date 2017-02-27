/*
 * Copyright (C) 2011-2012 by  Fernando Amat
 * See license.txt for full license and copyright notice.
 *
 * Authors: Fernando Amat 
 *
 * mainTest.cpp
 *
 *  Created on: September 17th, 2012
 *      Author: Fernando Amat
 *
 * \brief Generates a hierachical segmentation of a 3D stack and saves teh result in binary format
 *
 */


#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "watershedPersistanceAgglomeration.h"
#include "CUDAmedianFilter2D/medianFilter2D.h"
#ifdef PICTOOLS_JP2K
#include "ioFunctions.h"
#endif

#include "parseConfigFile.h"

namespace mylib
{
	#include "../temporalLogicalRules/mylib/array.h"
	#include "../temporalLogicalRules/mylib/image.h"
}

using namespace std;


void generateSegmentationMaskFromHS(string hsFilename, int tau, size_t minSvSize);
void parseImageFilePattern(string& imgRawPath, int frame);

int main( int argc, const char** argv )
{


	if( argc == 4)//we have a .bin file from hierarchical segmentation and we want to output a segmentation for a specific tau
	{
		//Call function with ProcessStack <hsFilename.bin> <tau> <minSupervoxelSize>

		string hsFilename(argv[1]);
		int tau = atoi(argv[2]);
		size_t minSvSize = atoi(argv[3]);

		generateSegmentationMaskFromHS(hsFilename, tau, minSvSize);

		return 0;
	}

	
	//parse input parameters
	string basename;//filename without extension so we can save our binary hierarchical segmentation
	int radiusMedianFilter = 0;//if radius = 1->3x3 medianFilter
	int minTau = 0;
	int backgroundThr = 0;
	int conn3D = 0;

	if( argc == 3 ) //we call program wiht <configFile> <timePoint>
	{
		configOptionsTrackingGaussianMixture configOptions;
		string configFilename(argv[1]);
		int err = configOptions.parseConfigFileTrackingGaussianMixture(configFilename);
		if( err != 0 ) 
			return err;

		int frame = atoi( argv[2] );
		basename = configOptions.imgFilePattern;
		parseImageFilePattern(basename, frame);

		radiusMedianFilter = configOptions.radiusMedianFilter;
		minTau = configOptions.minTau;
		backgroundThr = configOptions.backgroundThreshold;
		conn3D = configOptions.conn3D;

	}else if( argc == 6)
	{
		basename = string(argv[1]);//filename without extension so we can save our binary hierarchical segmentation
		radiusMedianFilter = atoi(argv[2]);//if radius = 1->3x3 medianFilter
		minTau = atoi(argv[3]);
		backgroundThr = atoi(argv[4]);
		conn3D = atoi(argv[5]);

	}else{
		cout<<"Wrong number of parameters. Call function with <imgBasename> <radiusMedianFilter> <minTau> <backgroundThr>  <conn3D>"<<endl;
		cout<<"Wrong number of parameters. Call function with <configFile> <frame>"<<endl;
		return 2;
	}
	

	int devCUDA = 0;
	//================================================================================
	
	//read current image
	string imgFilename(basename + ".tif");
	mylib::Array* img = mylib::Read_Image((char*)(imgFilename.c_str()),0);
	if( img == NULL )
	{		
#ifdef PICTOOLS_JP2K
		//try to read JP2 image
		imgFilename = string(basename + ".jp2");
		mylib::Value_Type type;
		int ndims;
		mylib::Dimn_Type  *dims = NULL;
		void* data = readJP2Kfile(imgFilename, type, ndims, &dims);


		if( data == NULL)
		{
			cout<<"ERROR: could not open file "<<imgFilename<<" to read input image"<<endl;
			return 5;
		}

		img = mylib::Make_Array_Of_Data(mylib::PLAIN_KIND, type, ndims, dims, data);
#else
		cout<<"ERROR: could not open file "<<imgFilename<<" to read input image"<<endl;
		return 5;
#endif
	}


	//hack to make the code work for uin8 without changing everything to templates
	//basically, parse the image to uint16, since the code was designed for uint16
	if( img->type == mylib::UINT8_TYPE )
	{	
		img = mylib::Convert_Array_Inplace (img, img->kind, mylib::UINT16_TYPE, 16, 0);
	}
	
	//hack to make the code work for 2D without changing everything to templates
	//basically, add one black slice to the image (you should select conn3D = 4 or 8)
	if( img->ndims == 2 )
	{	
		mylib::Dimn_Type dimsAux[dimsImage];
		for(int ii = 0; ii<img->ndims; ii++)
			dimsAux[ii] = img->dims[ii];
		for(int ii = img->ndims; ii<dimsImage; ii++)
			dimsAux[ii] = 2;

		mylib::Array *imgAux = mylib::Make_Array(img->kind, img->type, dimsImage, dimsAux);
		memset(imgAux->data,0, (imgAux->size) * sizeof(mylib::uint16) ); 
		memcpy(imgAux->data, img->data, img->size * sizeof(mylib::uint16) ); 
	
		mylib::Array* imgSwap = imgAux;
		img = imgAux;
		mylib::Free_Array( imgSwap);
	}

	if( img->type != mylib::UINT16_TYPE )
	{
		cout<<"ERROR: code is not ready for this types of images (change imgVoxelType and recompile)"<<endl;
		return 2;
	}
	
	
	//calculate median filter
	medianFilterCUDASliceBySlice((imgVoxelType*) (img->data), img->dims, radiusMedianFilter,devCUDA);

	//build hierarchical tree
	//cout<<"DEBUGGING: building hierarchical tree"<<endl;
	int64 imgDims[dimsImage];
	for(int ii = 0;ii<dimsImage; ii++)
		imgDims[ii] = img->dims[ii];
	hierarchicalSegmentation* hs =  buildHierarchicalSegmentation((imgVoxelType*) (img->data), imgDims, backgroundThr, conn3D, minTau, 1);
	
	//save hierarchical histogram
	//cout<<"DEBUGGING: saving hierarchical histogram"<<endl;
	char buffer[256];
	sprintf(buffer,"_hierarchicalSegmentation_conn3D%d_medFilRad%d",conn3D,radiusMedianFilter);
	string suffix(buffer);
	string fileOutHS(basename + suffix + ".bin");
	ofstream os(fileOutHS.c_str(), ios::binary | ios:: out);
	if( !os.is_open() )
	{
		cout<<"ERROR: could not open file "<< fileOutHS<<" to save hierarchical segmentation"<<endl;
		return 3;
	}
	hs->writeToBinary( os );
	os.close();

	//save parameters	
	fileOutHS = string(basename + suffix + ".txt");
	ofstream osTxt(fileOutHS.c_str());
	if( !osTxt.is_open() )
	{
		cout<<"ERROR: could not open file "<< fileOutHS<<" to save hierarchical segmentation paremeters"<<endl;
		return 3;
	}

	osTxt<<"Image basename = "<<basename<<endl;
	osTxt<<"Radius median filter = "<<radiusMedianFilter<<endl;
	osTxt<<"Min tau ="<<minTau<<endl;
	osTxt<<"Background threshold = "<<backgroundThr<<endl;
	osTxt<<"Conn3D = "<<conn3D<<endl;
	osTxt.close();

	//release memory
	delete hs;


	return 0;
}


//==================================================================================
void generateSegmentationMaskFromHS(string hsFilename, int tau, size_t minSvSize)
{

	//read hierarchical segmentation
	ifstream fin(hsFilename.c_str(), ios::binary | ios::in );

	if( fin.is_open() == false )
	{
		cout<<"ERROR: at generateSegmentationMaskFromHS: opening file "<<hsFilename<<endl;
		exit(4);
	}

	hierarchicalSegmentation* hs = new hierarchicalSegmentation(fin);
	fin.close();


	//generate a segmentation for the given tau
	hs->segmentationAtTau(tau);


	//generate array with segmentation mask
	mylib::Dimn_Type dimsVec[dimsImage];
	for(int ii = 0; ii <dimsImage; ii++)
		dimsVec[ii] = supervoxel::dataDims[ii];

	mylib::Array *imgL = mylib::Make_Array(mylib::PLAIN_KIND,mylib::UINT16_TYPE, dimsImage, dimsVec);
	mylib::uint16* imgLptr = (mylib::uint16*)(imgL->data);

	memset(imgLptr, 0, sizeof(mylib::uint16) * (imgL->size) );
	mylib::uint16 numLabels = 0;
	for(vector<supervoxel>::iterator iterS = hs->currentSegmentatioSupervoxel.begin(); iterS != hs->currentSegmentatioSupervoxel.end(); ++iterS)
	{
		if( iterS->PixelIdxList.size() < minSvSize )
			continue;
		if( numLabels == 65535 )
		{
			cout<<"ERROR: at generateSegmentationMaskFromHS: more labels than permitted in uint16"<<endl;
			exit(4);
		}

		numLabels++;
		for(vector<uint64>::iterator iter = iterS->PixelIdxList.begin(); iter != iterS->PixelIdxList.end(); ++iter)
		{
			imgLptr[ *iter ] = numLabels;
		}		
	}

	cout<<"A total of "<<numLabels<<" labels for tau="<<tau<<endl;

	//write tiff file
	char buffer[128];
	sprintf(buffer,"_tau%d",tau);
	string suffix(buffer);
	string imgLfilename( hsFilename +  suffix + ".tif");
	mylib::Write_Image( (char*) (imgLfilename.c_str()), imgL, mylib::DONT_PRESS );

	//write jp2 file
#ifdef PICTOOLS_JP2K
	imgLfilename = string( hsFilename +  suffix + ".jp2");
	writeJP2Kfile(imgL, imgLfilename);
#endif
	cout<<"Tiff file written to "<<imgLfilename<<endl;

	//release memory
	delete hs;
	mylib::Free_Array(imgL);
}



//==================================================================================
//=======================================================================
void parseImageFilePattern(string& imgRawPath, int frame)
{	

	size_t found=imgRawPath.find_first_of("?");
	while(found != string::npos)
	{
		int intPrecision = 0;
		while ((imgRawPath[found] == '?') && found != string::npos)
		{
			intPrecision++;
			found++;
			if( found >= imgRawPath.size() )
				break;

		}

		
		char bufferTM[16];
		switch( intPrecision )
		{
		case 2:
			sprintf(bufferTM,"%.2d",frame);
			break;
		case 3:
			sprintf(bufferTM,"%.3d",frame);
			break;
		case 4:
			sprintf(bufferTM,"%.4d",frame);
			break;
		case 5:
			sprintf(bufferTM,"%.5d",frame);
			break;
		case 6:
			sprintf(bufferTM,"%.6d",frame);
			break;
		}
		string itoaTM(bufferTM);
		
		found=imgRawPath.find_first_of("?");
		imgRawPath.replace(found, intPrecision,itoaTM);
		

		//find next ???
		found=imgRawPath.find_first_of("?");
	}
	
}
