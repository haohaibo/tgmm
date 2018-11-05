/*
*  EllipticalHaarFeatures.cpp
*
*     \brief Implements the functions in EllipticalHaarFeatures.h that are not
* CUDA related
*
*/

#include "EllipticalHaarFeatures.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>

using namespace std;

bool basicEllipticalHaarFeatureVector::useDoGfeatures = true;

void generateHaarPattern(int **Haar, int numRings) {
  int *symbols = new int[numRings];  // keeps the counter for each ring

  for (int ii = 0; ii < numRings; ii++) symbols[ii] = -1;

  symbols[0] = 0;  // the most signigicant bit can be strted at 0 (all the -1
                   // are nbot needed)

  // addition for a terciary variable {-1,0,1}
  int pos = 0;
  int N = 0;
  bool exitFlag = false;
  bool zeroMinusFlag = true;
  while (exitFlag == false) {
    // check for 0...0,-1 pattern or for all zeros
    zeroMinusFlag = false;
    pos = 0;
    while (symbols[pos] == 0) {
      if (pos == numRings - 1) {
        zeroMinusFlag = true;  // all zeros pattern
        break;
      }
      pos++;
    }
    if (symbols[pos] == -1) zeroMinusFlag = true;

    // symbol can be updated
    if (zeroMinusFlag == false) {
      memcpy(Haar[N++], symbols, sizeof(int) * numRings);
    }

    // increment the counter
    pos = numRings - 1;
    symbols[pos]++;
    while (symbols[pos] > 1) {
      symbols[pos] = -1;
      pos--;
      if (pos < 0) {
        exitFlag = true;
        break;
      }
      symbols[pos]++;
    }
  }
  delete[] symbols;
}

int getNumberOfHaarFeaturesPerllipsoid() {
  // precalculate number of HaarFeaturesPerEllipsoid
  // VIP: IF YOU CHANGE THE CODE, YOU HAVE TO CHANGE THIS CALCULATION
  basicEllipticalHaarFeatureVector *fBasic =
      new basicEllipticalHaarFeatureVector();
  int numCells = fBasic->numCells;
  int numRings = fBasic->numRings;

  int nchoose2Aux = numCells * (numCells - 1) / 2;
  int tripletAux = (((int)pow(3.0f, numRings)) + 1) / 2 -
                   1;  // number of linearly independent combinations of {1,0,1}
                       // for each ring. (0,0,0) is exlcuded, thus, the -1.

  delete fBasic;

  if (basicEllipticalHaarFeatureVector::useDoGfeatures == true)
    return ((numCells + tripletAux + nchoose2Aux +
             nchoose2Aux * (nchoose2Aux - 1) / 2) *
                2 +
            dimsImage * (dimsImage - 1) / 2);
  else
    return ((numCells + tripletAux + nchoose2Aux +
             nchoose2Aux * (nchoose2Aux - 1) / 2) +
            dimsImage * (dimsImage - 1) / 2);
}

void calculateCombinationsOfBasicHaarFeatures(
    basicEllipticalHaarFeatureVector **fBasic, int numEllipsoids,
    int *numHaarFeaturesPerEllipsoid, float **fVec_) {
  if (numEllipsoids <= 0) {
    if (*fVec_ != NULL) {
      delete (*fVec_);
      *fVec_ = NULL;
    }
    return;
  }

  // precalculate number of HaarFeaturesPerEllipsoid
  // VIP: IF YOU CHANGE THE CODE, YOU HAVE TO CHANGE THIS CALCULATION
  *numHaarFeaturesPerEllipsoid = getNumberOfHaarFeaturesPerllipsoid();

  int numCells = fBasic[0]->numCells;
  int numRings = fBasic[0]->numRings;

  int nchoose2Aux = numCells * (numCells - 1) / 2;
  int tripletAux = (((int)pow(3.0f, numRings)) + 1) / 2 -
                   1;  // number of linearly independent combinations of {1,0,1}
                       // for each ring. (0,0,0) is exlcuded, thus, the -1.

  // allocate memory
  if ((*fVec_) == NULL)
    (*fVec_) =
        new float[((long long int)numEllipsoids) *
                  ((long long int)(*numHaarFeaturesPerEllipsoid))];  // this can
                                                                     // be
                                                                     // larger
                                                                     // than 2GB

  float *fVec = (*fVec_);

  float *fPairAdd = new float[nchoose2Aux];  // to store intermediate for last
                                             // calculations between differences
                                             // of pairs of additions

  // generate some combinatorial elements (Haar {-1,0,1}) that we will need to
  // combinae Haar features
  int **HaarPatternRings = new int *[tripletAux];
  for (int ii = 0; ii < tripletAux; ii++) {
    HaarPatternRings[ii] = new int[numRings];
    memset(HaarPatternRings[ii], 0, sizeof(int) * numRings);
  }
  generateHaarPattern(HaarPatternRings, numRings);

  // extend Haar features for each ellipsoid
  basicEllipticalHaarFeatureVector *fAux = NULL;
  long long int numFeatures = 0;
  for (int ii = 0; ii < numEllipsoids; ii++) {
    fAux = fBasic[ii];

    // excentricity
    for (int jj = 0; jj < dimsImage; jj++)
      fVec[numFeatures++] = fAux->excentricity[jj];

    // average intensity on each cell
    for (int jj = 0; jj < numCells; jj++)
      fVec[numFeatures++] = fAux->cellAvgIntensity[jj];

    // all possible combinations {-1,0,+1} for each ring average intensity
    for (int ii = 0; ii < tripletAux; ii++) {
      fVec[numFeatures] = 0;
      for (int jj = 0; jj < numRings; jj++) {
        fVec[numFeatures] +=
            ((float)HaarPatternRings[ii][jj]) * fAux->ringAvgIntensity[jj];
      }
      numFeatures++;
    }

    // all possible differences between cells
    // I also precompute all possible pairs of additions so I can compute the
    // differences faster later on
    int count = 0;
    float auxFval = 0.0f;
    for (int ii = 0; ii < numCells; ii++) {
      auxFval = fAux->cellAvgIntensity[ii];
      for (int jj = ii + 1; jj < numCells; jj++) {
        fVec[numFeatures++] = auxFval - fAux->cellAvgIntensity[jj];
        fPairAdd[count++] = auxFval + fAux->cellAvgIntensity[jj];
      }
    }

    // calculate all possible differences between pair of additions
    for (int ii = 0; ii < nchoose2Aux; ii++) {
      auxFval = fPairAdd[ii];
      for (int jj = ii + 1; jj < nchoose2Aux; jj++) {
        fVec[numFeatures++] = auxFval - fPairAdd[jj];
      }
    }

    //====================================================================================
    //==============================copy and paste code for
    // DoG===============================
    //=====================================================================================
    if (basicEllipticalHaarFeatureVector::useDoGfeatures == true) {
      // average intensity on each cell
      for (int jj = 0; jj < numCells; jj++)
        fVec[numFeatures++] = fAux->cellAvgIntensityDoG[jj];

      // all possible combinations {-1,0,+1} for each ring average intensity
      for (int ii = 0; ii < tripletAux; ii++) {
        fVec[numFeatures] = 0;
        for (int jj = 0; jj < numRings; jj++) {
          fVec[numFeatures] +=
              ((float)HaarPatternRings[ii][jj]) * fAux->ringAvgIntensityDoG[jj];
        }
        numFeatures++;
      }

      // all possible differences between cells
      // I also precompute all possible pairs of additions so I can compute the
      // differences faster later on
      count = 0;
      auxFval = 0.0f;
      for (int ii = 0; ii < numCells; ii++) {
        auxFval = fAux->cellAvgIntensityDoG[ii];
        for (int jj = ii + 1; jj < numCells; jj++) {
          fVec[numFeatures++] = auxFval - fAux->cellAvgIntensityDoG[jj];
          fPairAdd[count++] = auxFval + fAux->cellAvgIntensityDoG[jj];
        }
      }

      // calculate all possible differences between pair of additions
      for (int ii = 0; ii < nchoose2Aux; ii++) {
        auxFval = fPairAdd[ii];
        for (int jj = ii + 1; jj < nchoose2Aux; jj++) {
          fVec[numFeatures++] = auxFval - fPairAdd[jj];
        }
      }
    }
  }

  if (numFeatures !=
      ((long long int)(*numHaarFeaturesPerEllipsoid)) *
          ((long long int)numEllipsoids)) {
    delete[](*fVec_);
    (*fVec_) = NULL;
    cout << "ERROR: at calculateCombinationsOfBasicHaarFeatures the number of "
            "features does not agree with the expected value. Revise the code "
            "to make sure the precalculation is correct"
         << endl;
    return;
  }

  // release memory
  for (int ii = 0; ii < tripletAux; ii++) delete[] HaarPatternRings[ii];
  delete[] HaarPatternRings;
  delete[] fPairAdd;

  return;
}
