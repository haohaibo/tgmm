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
 *  Modified By: Haibo Hao
 *  Date: March 31th, 2017
 *  Email: haohaibo@ncic.ac.cn
 *
 * \brief Shows how to use the watershedPersistanceAgglomeration.cpp functions
 *
 */


#include <string>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "watershedPersistanceAgglomeration.h"
namespace medianFilter2D //to perform median filter slice by slice
{
	#include "external/medianFilter2D/ctmf.h"
}

using namespace std;
//---------------------------------------------------
//--forward definition of the different test functions--
//test watershed+merging algorithm
int mainTestBasic( int argc, const char** argv );
//counts average number of neighbors per voxel in a 
//segmented region. It could be useful to differentiate 
//foreground / background regions
int mainTestNumberOfAvgNeighbors(int argc, const char** argv );
//test hierarchical segmentation (the segmentation for all tau at once)
int mainTestHierarchicalSegmentation( int argc, const char** argv );
//---------------------------------------------------

//main function
int main( int argc, const char** argv )
{
	//return mainTestBasic(argc, argv);
	//return mainTestNumberOfAvgNeighbors(argc, argv);
	return mainTestHierarchicalSegmentation(argc, argv);
}


//==========================================================
int mainTestBasic( int argc, const char** argv )
{
	
	int radiusMedianFilter = 3; //radiu sof teh emdian filter. A 3x3 median filter has radius 1
	
	//-------------small size---------------
	
	string filenameIn("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7_PS.bin");
	int64 imgDims[dimsImage] = {441, 261, 50};
	int conn3D = 74;
	imgVoxelType backgroundThr = 100;
	imgVoxelType tau = 10;
	

	
	//-------------medium size (drosophila)---------------
	/*
	string filenameIn("G:\\12-07-17\\TimeFused\\Dme_E1_SpiderGFP-His2ARFP.TM0000_timeFused_blending\\SPC0_TM0000_CM0_CM1_CHN00_CHN01.fusedStack_medFilt5.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7_PS.bin");
	int64 imgDims[dimsImage] = {1292,630,123};
	int conn3D = 74;
	imgVoxelType backgroundThr = 200;
	imgVoxelType tau = 14;
	*/

	
	//-------------large size (sebrafish)---------------
	/*
	string filenameIn("E:\\12-06-07\\Dre_E1_H2BeGFP_01_20120607_200041.corrected\\TimeFused\\Dre_E1_H2BeGFP.TM00000_timeFused_blending\\CM0_CM1_CHN00_CHN01.fusedStack_00000_medFilt5.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7_PS.bin");
	int64 imgDims[dimsImage] = {1740, 1800, 269};
	int conn3D = 74;
	imgVoxelType backgroundThr = 100;
	imgVoxelType tau = 40;
	*/

	//allocate memory
	int64 imgSize = 1;
	for(int ii =0;ii<dimsImage;ii++) imgSize *= imgDims[ii];
	imgVoxelType *img = new imgVoxelType[imgSize];
	imgLabelType *imgL = new imgLabelType[imgSize];

	//read MRC file
	FILE *fid = fopen(filenameIn.c_str(),"rb");
	if( fid == NULL)
	{
		cout<<"ERROR: opening file " << filenameIn<<endl;
		return 3;
	}
	fseek(fid,1024,SEEK_SET);
	fread(img,sizeof(imgVoxelType), imgSize,fid);
	fclose(fid);

	//calculate median filter
	//TODO

	//call watershed
	cout<<"Calculating watershed persistance agglomeration..."<<endl;
	imgLabelType numLabels;
	time_t start, end;
	time(&start);
	//int err = watershedPersistanceAgglomeration(img, imgDims, backgroundThr, conn3D, tau, imgL, &numLabels);
	int err = watershedPersistanceAgglomerationMultithread(img, imgDims, backgroundThr, conn3D, tau, imgL, &numLabels, 10);
	time(&end);
	if (err > 0)
		return err;
	cout<<"Partitioned the image in "<<numLabels<<" regions took "<<difftime(end,start)<<" secs"<<endl;

	//save result
	cout<<"Saving results in file "<<filenameOut<<endl;
	FILE *fout = fopen(filenameOut.c_str(),"wb");
	if( fout == NULL)
	{
		cout<<"ERROR: opening file " << filenameOut<<endl;
		return 3;
	}
	fwrite(imgL,sizeof(imgLabelType),imgSize,fout);
	fclose(fout);

	//release memory
	delete[] img;
	delete[] imgL;

	return 0;
}


//=====================================================================================
int mainTestNumberOfAvgNeighbors(int argc, const char** argv )
{
	
	int conn3D = 4;

	
	string imgLfilename("G:\\12-07-17\\TimeFused_BackgrSubtraction_thrPctile40_maxSize3000_otzu\\TM00039\\CM0_CM1_CHN00_CHN01.fusedStack_bckgSub_PersistanceSeg_tau14_00039.bin");
	int64 imgDims[dimsImage] = {630, 1292, 123};
	

	cout<<"Testing mainTestNumberOfAvgNeighbors with conn3D="<<conn3D<<endl;

	//allocate memory
	int64 imgSize = 1;
	for(int ii =0;ii<dimsImage;ii++) 
		imgSize *= imgDims[ii];
		
	imgLabelType *imgL = new imgLabelType[imgSize];

	//read binary file
	FILE *fid = fopen( imgLfilename.c_str(),"rb");
	if( fid == NULL)
	{
		cout<<"ERROR: opening file " << imgLfilename<<endl;
		return 3;
	}
	fread(imgL,sizeof(imgLabelType), imgSize,fid);
	fclose(fid);


	//calculate average number of neighbors per region
	vector<float> avgNumberOfNeighbors;
	avgNumberOfNeighbors.reserve(10000);

	time_t start, end;
	time(&start);
	int err = averageNumberOfNeighborsPerRegion(imgL,imgDims, conn3D, avgNumberOfNeighbors);
	if(err > 0)
		return err;
	time(&end);
	cout<<"It took "<<difftime(end, start)<<" secs "<<endl;

	//save results
	float* imgN = new float[imgSize];
	for(int64 ii =0;ii < imgSize; ii++)
		imgN[ii] = avgNumberOfNeighbors[ imgL[ii] ];

	//write out file
	string outFilename( imgLfilename + "_avgNumNeigh.bin");
	fid = fopen( outFilename.c_str(),"wb");
	if( fid == NULL)
	{
		cout<<"ERROR: opening file " << outFilename<<endl;
		return 3;
	}
	fwrite(imgN, sizeof(float), imgSize, fid);
	fclose(fid);
	cout<<"Writen solution in file "<<outFilename<<" in float32 precision with size "<<imgDims[0]<<"x"<<imgDims[1]<<"x"<<imgDims[2]<<endl;

	ofstream outTxt( (imgLfilename + "_avgNumNeigh.txt").c_str() );
	for(size_t ii =0; ii < avgNumberOfNeighbors.size(); ii++)
		outTxt<<avgNumberOfNeighbors[ii]<<" ";
	outTxt.close();

	//release memory
	delete[] imgL;
	delete[] imgN;

	return 0;
}


//===========================================================================================
int mainTestHierarchicalSegmentation( int argc, const char** argv )
{
	cout<<"TESTING mainTestHierarchicalSegmentation"<<endl;
	int radiusMedianFilter = 3; //radius of the mdian filter. A 3x3 median filter has radius 1
	//string fileOutHS("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testHS.hs");//file to save hierarchical histogram
    
    //file to save hierarchical histogram
	string fileOutHS("/home/hhb/work/tgmm-hhb-m/nmeth.3036-S2/data/test/testHS.hs");
	
	//-------------synthetic line---------------
	/*
	string filenameIn("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\syntheticLine.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\syntheticLine_PS.bin");
	int64 imgDims[dimsImage] = {20, 601, 10};
	int conn3D = 74;
	imgVoxelType backgroundThr = 10;
	imgVoxelType tau = 10;//10->3 regions; 30->2 regions; 40->2 regions; 70->1 regions
	imgVoxelType minTau = 2;
	*/
	//-------------small size---------------
	
	string filenameIn("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7_PS.bin");
	int64 imgDims[dimsImage] = {441, 261, 50};
	int conn3D = 74;
	imgVoxelType backgroundThr = 100;
	imgVoxelType tau = 22;
	imgVoxelType minTau = 2;
	
	
	//-------------medium size (drosophila)---------------
	/*
	string filenameIn("G:\\12-07-17\\TimeFused\\Dme_E1_SpiderGFP-His2ARFP.TM00000_timeFused_blending\\SPC0_TM0000_CM0_CM1_CHN00_CHN01.fusedStack_medFilt5.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7_PS.bin");
	int64 imgDims[dimsImage] = {1292,630,123};
	int conn3D = 74;
	imgVoxelType backgroundThr = 200;
	imgVoxelType tau = 14;
	imgVoxelType minTau = 2;
	*/
	
	//-------------large size (sebrafish)---------------
	/*
	string filenameIn("E:\\12-06-07\\Dre_E1_H2BeGFP_01_20120607_200041.corrected\\TimeFused\\Dre_E1_H2BeGFP.TM00000_timeFused_blending\\CM0_CM1_CHN00_CHN01.fusedStack_00000_medFilt5.mrc");
	string filenameOut("C:\\Users\\Fernando\\cppProjects\\nucleiChannelSupervoxelsWatershedWithPersistanceAgglomeration\\testData\\testVol_medFilt7_PS.bin");
	int64 imgDims[dimsImage] = {1740, 1800, 269};
	int conn3D = 74;
	imgVoxelType backgroundThr = 100;
	imgVoxelType tau = 40;
	*/

	//allocate memory
	int64 imgSize = 1;
	for(int ii =0;ii<dimsImage;ii++) imgSize *= imgDims[ii];
	imgVoxelType *img = new imgVoxelType[imgSize];
	imgLabelType *imgL = new imgLabelType[imgSize];

	//read MRC file
	FILE *fid = fopen(filenameIn.c_str(),"rb");
	if( fid == NULL)
	{
		cout<<"ERROR: opening file " << filenameIn<<endl;
		return 3;
	}
	fseek(fid,1024,SEEK_SET);
	fread(img,sizeof(imgVoxelType), imgSize,fid);
	fclose(fid);

	time_t start, end;

	//calculate median filter
	//TODO

	//build hierarchical tree
	time(&start);
	hierarchicalSegmentation* hs =  buildHierarchicalSegmentation(img, imgDims, backgroundThr, conn3D, minTau, 8);
	time(&end);
	cout<<"Entire function buildHierarchicalSegmentation took "<<difftime(end,start)<< " secs"<<endl;

	if( hs->debugCheckDendrogramCoherence() > 0 )
		exit(3);

	//generate a segmentation for the given tau
	time(&start);
	hs->segmentationAtTau(tau);
	time(&end);
	cout<<"Entire function segmentationAtTau took "<<difftime(end,start)<< " secs to generate "<<hs->currentSegmentatioSupervoxel.size()<<" labels"<<endl;


	//compare with regular code for a single tau
	cout<<"Calculating watershed persistance agglomeration (a single tau)..."<<endl;
	imgLabelType numLabels;
	time(&start);
	int err = watershedPersistanceAgglomeration(img, imgDims, backgroundThr, conn3D, tau, imgL, &numLabels);
	time(&end);
	if (err > 0)
		return err;
	cout<<"Partitioned the image in "<<numLabels<<" regions took "<<difftime(end,start)<<" secs"<<endl;

	//perform comparison between one and the other
	cout<<"performing comparison between segmentations"<<endl;
	int *hist = new int[numLabels + 1];
	int numUniqueAssignment = 0;
	for(vector<supervoxel>::iterator iterS = hs->currentSegmentatioSupervoxel.begin(); iterS != hs->currentSegmentatioSupervoxel.end(); ++iterS)
	{
		memset(hist, 0, sizeof(int) * (numLabels +1) );
		for(vector<uint64>::iterator iter = iterS->PixelIdxList.begin(); iter != iterS->PixelIdxList.end(); ++iter)
		{
			hist[ imgL[*iter] ]++;
		}

		//check if it is encompased in a single label
		int maxVal = 0;
		for(imgLabelType ii = 0; ii<numLabels +1; ii++)
		{
			if( hist[ii] > maxVal )
				maxVal = hist[ii];
		}

		if( ((float) maxVal )/ ((float) iterS->PixelIdxList.size()) > 0.9 )
			numUniqueAssignment++;
	}
	cout<<numUniqueAssignment<<" labels out of "<<hs->currentSegmentatioSupervoxel.size()<<" have unique assignments"<<endl;


	//save hierarchical histogram
	cout<<"Saving hierarchical histogram into "<<fileOutHS<<endl;
	ofstream os(fileOutHS.c_str(), ios::binary | ios:: out);
	if( !os.is_open() )
	{
		cout<<"ERROR: could not open file "<< fileOutHS<<" to write"<<endl;
		exit(3);
	}
	hs->writeToBinary( os );
	os.close();

	//testing if I can read and perform the same operation
	cout<<"Testing hierarchical histogram reading... "<<endl;
	ifstream is( fileOutHS.c_str(), ios::binary | ios::out );
	hierarchicalSegmentation* hsRead = new hierarchicalSegmentation(is);
	is.close();

	hsRead->segmentationAtTau(tau);

	numUniqueAssignment = 0;
	for(vector<supervoxel>::iterator iterS = hsRead->currentSegmentatioSupervoxel.begin(); iterS != hsRead->currentSegmentatioSupervoxel.end(); ++iterS)
	{
		memset(hist, 0, sizeof(int) * (numLabels +1) );
		for(vector<uint64>::iterator iter = iterS->PixelIdxList.begin(); iter != iterS->PixelIdxList.end(); ++iter)
		{
			hist[ imgL[*iter] ]++;
		}

		//check if it is encompased in a single label
		int maxVal = 0;
		for(imgLabelType ii = 0; ii<numLabels +1; ii++)
		{
			if( hist[ii] > maxVal )
				maxVal = hist[ii];
		}

		if( ((float) maxVal )/ ((float) iterS->PixelIdxList.size()) > 0.9 )
			numUniqueAssignment++;
	}
	cout<<numUniqueAssignment<<" labels out of "<<hsRead->currentSegmentatioSupervoxel.size()<<" have unique assignments"<<endl;


	//release memory
	delete hs;
	delete hsRead;
	delete[] hist;
	delete[] img;
	delete[] imgL;

	return 0;
}
