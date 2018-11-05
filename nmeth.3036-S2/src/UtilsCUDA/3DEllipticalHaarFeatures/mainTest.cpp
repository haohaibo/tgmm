#include <fstream>
#include <iostream>
#include "EllipticalHaarFeatures.h"
#include "external/Nathan/tictoc.h"

using namespace std;
int main(int argc, const char **argv) {
  int devCUDA = 0;
  double *m = NULL;
  double *W = NULL;
  int numEllipsoids = 0;
  imageType *im = NULL;
  long long int dims[dimsImage];

  //-----------------test case 1----------------------------

  numEllipsoids = 1000;  // max of a 1000 for the file test
  m = new double[numEllipsoids * dimsImage];
  ifstream fm("E:/temp/CM0_CM1_CHN00_CHN01.fusedStack_00380_m.txt");
  for (int ii = 0; ii < numEllipsoids; ii++) {
    fm >> m[ii] >> m[ii + numEllipsoids] >>
        m[ii + numEllipsoids * 2];  // coalescent access for GPU
  }
  fm.close();

  W = new double[numEllipsoids * dimsImage * (1 + dimsImage) / 2];
  ifstream fW("E:/temp/CM0_CM1_CHN00_CHN01.fusedStack_00380_W.txt");
  for (int ii = 0; ii < numEllipsoids; ii++) {
    fW >> W[ii] >> W[ii + numEllipsoids] >> W[ii + numEllipsoids * 2] >>
        W[ii + numEllipsoids * 3] >> W[ii + numEllipsoids * 4] >>
        W[ii + numEllipsoids * 5];  // coalescent access for GPU
  }
  fW.close();

  dims[0] = 587;
  dims[1] = 1399;
  dims[2] = 112;
  long long int imSize = dims[0] * dims[1] * dims[2];

  im = new imageType[imSize];
  FILE *fid = fopen("E:/temp/CM0_CM1_CHN00_CHN01.fusedStack_00380.bin", "rb");
  fread(im, sizeof(imageType), imSize, fid);
  fclose(fid);

  //--------------------end of test case 1-----------------------

  //--------------test case 2 (synthetic image)--------------------
  /*
  numEllipsoids = 1;
  m = new double[numEllipsoids*dimsImage];
  m[0]=39.0144;m[1]=39.0037;m[2]=5.62859;
  W = new double[numEllipsoids*dimsImage*(1+dimsImage)/2];
  W[0] =109.83*0.00129047;W[1] = 109.83*3.75697e-006;W[2] = 109.83*0;W[3] =
  109.83*0.000632087;W[4] = 109.83*0;W[5] = 109.83*0.0267087;

  dims[0] = 96; dims[1] = 128; dims[2] = 26;
  long long int imSize=dims[0]*dims[1]*dims[2];

  im = new imageType[imSize];
  FILE* fid=fopen("E:/temp/synthetic_TM00000.bin","rb");
  fread(im,sizeof(imageType),imSize,fid);
  fclose(fid);
  */
  //--------------------end of test case 2-----------------------

  // call main function
  TicTocTimer timerF = tic();
  basicEllipticalHaarFeatureVector **fBasic =
      calculateEllipticalHaarFeatures(m, W, numEllipsoids, im, dims, devCUDA,
                                      0);  // no symmetry used in this example
  if (fBasic == NULL) return 1;            // some error happened

  // extend Haar features
  int numHaarFeaturesPerEllipsoid = 0;
  float *fVec = NULL;
  calculateCombinationsOfBasicHaarFeatures(
      fBasic, numEllipsoids, &numHaarFeaturesPerEllipsoid,
      &fVec);                  // function allocates fVec if it is NULL
  if (fVec == NULL) return 2;  // some error happened
  /*
  ofstream outF("E:/temp/CM0_CM1_CHN00_CHN01.fusedStack_00380_F.txt");
  for(int ii=0;ii<numEllipsoids;ii++)
          fBasic[ii]->basicEllipticalHaarFeatureVectorPrint(outF);
  outF.close();
  */
  /*
  ofstream outF("E:/temp/CM0_CM1_CHN00_CHN01.fusedStack_00380_Fextended.txt");
  long long int count = 0;
  for(int ii=0;ii<numEllipsoids;ii++)
  {
          for(int jj=0;jj<numHaarFeaturesPerEllipsoid;jj++)
                  outF<<fVec[count++]<<" ";

          outF<<endl;
  }
  outF.close();
  */
  delete[] m;
  delete[] W;
  delete[] im;
  for (int ii = 0; ii < numEllipsoids; ii++) delete fBasic[ii];
  delete[] fBasic;
  delete[] fVec;

  cout << "Program finished calculating Elliptical Haar features normally. It "
          "took "
       << toc(&timerF) << " secs for " << numEllipsoids << " ellipsoids and "
       << numHaarFeaturesPerEllipsoid << " extended features" << endl;
  return 0;
}